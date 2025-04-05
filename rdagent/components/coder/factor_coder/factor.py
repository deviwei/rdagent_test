from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from filelock import FileLock

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.core.exception import CodeFormatError, CustomRuntimeError, NoOutputError
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash


class FactorTask(CoSTEERTask):
    # TODO:  generalized the attributes into the Task
    # - factor_* -> *
    def __init__(
        self,
        factor_name,
        factor_description,
        factor_formulation,
        *args,
        variables: dict = {},
        resource: str = None,
        factor_implementation: bool = False,
        **kwargs,
    ) -> None:
        self.factor_name = (
            factor_name  # TODO: remove it in the later version. Keep it only for pickle version compatibility
        )
        self.factor_formulation = factor_formulation
        self.variables = variables
        self.factor_resources = resource
        self.factor_implementation = factor_implementation
        super().__init__(name=factor_name, description=factor_description, *args, **kwargs)

    @property
    def factor_description(self):
        """for compatibility"""
        return self.description

    def get_task_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""

    def get_task_information_and_implementation_result(self):
        return {
            "factor_name": self.factor_name,
            "factor_description": self.factor_description,
            "factor_formulation": self.factor_formulation,
            "variables": str(self.variables),
            "factor_implementation": str(self.factor_implementation),
        }

    @staticmethod
    def from_dict(dict):
        return FactorTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"


class FactorFBWorkspace(FBWorkspace):
    """
    用于通过文件方式实现因子的工作空间类。
    输入数据和输出因子值都通过文件方式进行读写。
    继承自 FBWorkspace（文件基础工作空间）。
    """

    # 定义执行状态的常量字符串
    FB_EXEC_SUCCESS = "Execution succeeded without error."  # 执行成功的状态消息
    FB_CODE_NOT_SET = "code is not set."  # 代码未设置的状态消息
    FB_EXECUTION_SUCCEEDED = "Execution succeeded without error."  # 执行成功的状态消息
    FB_OUTPUT_FILE_NOT_FOUND = "\nExpected output file not found."  # 输出文件未找到的状态消息
    FB_OUTPUT_FILE_FOUND = "\nExpected output file found."  # 输出文件已找到的状态消息

    def __init__(
        self,
        *args,
        raise_exception: bool = False,  # 控制是否抛出异常的标志
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)  # 调用父类初始化方法
        self.raise_exception = raise_exception  # 设置异常处理标志

    def hash_func(self, data_type: str = "Production") -> str:
        """
        生成缓存键的哈希函数
        :param data_type: 数据类型，默认为"Debug"
        :return: 返回因子代码的MD5哈希值或None
        """
        return (
            md5_hash(data_type + self.file_dict["factor.py"])  # 如果存在factor.py且不抛出异常，返回其哈希值
            if ("factor.py" in self.file_dict and not self.raise_exception)
            else None  # 否则返回None
        )

    @cache_with_pickle(hash_func)  # 使用pickle缓存装饰器，缓存键由hash_func生成
    def execute(self, data_type: str = "Production") -> Tuple[str, pd.DataFrame]:
        """执行因子实现代码并获取因子值"""
        self.before_execute()  # 执行前的准备工作
        
        # 检查代码文件是否存在
        if self.file_dict is None or "factor.py" not in self.file_dict:
            if self.raise_exception:
                raise CodeFormatError(self.FB_CODE_NOT_SET)  # 如果需要抛出异常则抛出
            else:
                return self.FB_CODE_NOT_SET, None  # 否则返回错误信息
        
        with FileLock(self.workspace_path / "execution.lock"):  # 使用文件锁确保并发安全
            # 根据版本选择数据源路径
            if self.target_task.version == 1:
                source_data_path = (
                    Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)  # Debug模式使用debug数据文件夹
                    if data_type == "Debug"
                    else Path(FACTOR_COSTEER_SETTINGS.data_folder)  # 否则使用正常数据文件夹
                )
            elif self.target_task.version == 2:
                # Kaggle比赛数据路径
                source_data_path = Path(KAGGLE_IMPLEMENT_SETTING.local_data_path) / KAGGLE_IMPLEMENT_SETTING.competition

            source_data_path.mkdir(exist_ok=True, parents=True)  # 创建数据源目录
            code_path = self.workspace_path / f"factor.py"  # 设置代码文件路径

            # 链接所有数据文件到工作空间
            self.link_all_files_in_folder_to_workspace(source_data_path, self.workspace_path)

            execution_feedback = self.FB_EXECUTION_SUCCEEDED  # 初始化执行反馈
            execution_success = False  # 初始化执行状态
            execution_error = None  # 初始化错误信息

            # 根据版本选择执行代码路径
            if self.target_task.version == 1:
                execution_code_path = code_path  # 版本1直接使用factor.py
            elif self.target_task.version == 2:
                execution_code_path = self.workspace_path / f"{uuid.uuid4()}.py"  # 版本2生成临时执行文件
                execution_code_path.write_text((Path(__file__).parent / "factor_execution_template.txt").read_text())

            try:
                # 执行因子代码
                subprocess.check_output(
                    f"{FACTOR_COSTEER_SETTINGS.python_bin} {execution_code_path}",
                    shell=True,
                    cwd=self.workspace_path,
                    stderr=subprocess.STDOUT,
                    timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
                )
                execution_success = True  # 执行成功
            except subprocess.CalledProcessError as e:
                # 处理执行错误
                import site
                execution_feedback = (
                    e.output.decode()
                    .replace(str(execution_code_path.parent.absolute()), r"/path/to")
                    .replace(str(site.getsitepackages()[0]), r"/path/to/site-packages")
                )
                # 截断过长的错误信息
                if len(execution_feedback) > 2000:
                    execution_feedback = (
                        execution_feedback[:1000] + "....hidden long error message...." + execution_feedback[-1000:]
                    )
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)  # 抛出异常
                else:
                    execution_error = CustomRuntimeError(execution_feedback)  # 记录错误
            except subprocess.TimeoutExpired:
                # 处理超时错误
                execution_feedback += f"Execution timeout error and the timeout is set to {FACTOR_COSTEER_SETTINGS.file_based_execution_timeout} seconds."
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)  # 抛出异常
                else:
                    execution_error = CustomRuntimeError(execution_feedback)  # 记录错误

            # 检查并读取输出文件
            workspace_output_file_path = self.workspace_path / "result.h5"
            if workspace_output_file_path.exists() and execution_success:
                try:
                    # 读取因子值数据
                    executed_factor_value_dataframe = pd.read_hdf(workspace_output_file_path)
                    execution_feedback += self.FB_OUTPUT_FILE_FOUND
                except Exception as e:
                    # 处理读取错误
                    execution_feedback += f"Error found when reading hdf file: {e}"[:1000]
                    executed_factor_value_dataframe = None
            else:
                # 处理输出文件不存在的情况
                execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                executed_factor_value_dataframe = None
                if self.raise_exception:
                    raise NoOutputError(execution_feedback)  # 抛出异常
                else:
                    execution_error = NoOutputError(execution_feedback)  # 记录错误

        return execution_feedback, executed_factor_value_dataframe  # 返回执行结果和因子值

    def __str__(self) -> str:
        """返回工作空间的字符串表示"""
        return f"File Factor[{self.target_task.factor_name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        """返回工作空间的开发者字符串表示"""
        return self.__str__()

    @staticmethod
    def from_folder(task: FactorTask, path: Union[str, Path], **kwargs):
        """
        从文件夹创建工作空间实例的静态方法
        :param task: 因子任务对象
        :param path: 文件夹路径
        :return: 返回工作空间实例
        """
        path = Path(path)  # 转换路径为Path对象
        code_dict = {}  # 初始化代码字典
        for file_path in path.iterdir():  # 遍历文件夹
            if file_path.suffix == ".py":  # 只处理Python文件
                code_dict[file_path.name] = file_path.read_text()  # 读取文件内容
        return FactorFBWorkspace(target_task=task, code_dict=code_dict, **kwargs)  # 创建并返回工作空间实例


FactorExperiment = Experiment
FeatureExperiment = Experiment
