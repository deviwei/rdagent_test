# 导入路径处理模块
from pathlib import Path
# 导入类型提示模块
from typing import Any

# 导入命令行参数处理库
import fire

# 导入数据科学相关配置
from rdagent.app.data_science.conf import DS_RD_SETTING
# 导入集成学习相关组件
from rdagent.components.coder.data_science.ensemble import EnsembleCoSTEER
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
# 导入特征工程相关组件
from rdagent.components.coder.data_science.feature import FeatureCoSTEER
from rdagent.components.coder.data_science.feature.exp import FeatureTask
# 导入模型相关组件
from rdagent.components.coder.data_science.model import ModelCoSTEER
from rdagent.components.coder.data_science.model.exp import ModelTask
# 导入数据加载相关组件
from rdagent.components.coder.data_science.raw_data_loader import DataLoaderCoSTEER
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
# 导入工作流相关组件
from rdagent.components.coder.data_science.workflow import WorkflowCoSTEER
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
# 导入基础配置
from rdagent.components.workflow.conf import BasePropSetting
# 导入研发循环基类
from rdagent.components.workflow.rd_loop import RDLoop
# 导入异常处理类
from rdagent.core.exception import CoderError, RunnerError
# 导入实验反馈类
from rdagent.core.proposal import ExperimentFeedback
# 导入场景类
from rdagent.core.scenario import Scenario
# 导入工具函数
from rdagent.core.utils import import_class
# 导入日志工具
from rdagent.log import rdagent_logger as logger
# 导入数据科学相关组件
from rdagent.scenarios.data_science.dev.feedback import DSExperiment2Feedback
from rdagent.scenarios.data_science.dev.runner import DSCoSTEERRunner
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSExpGen, DSTrace
# 导入 Kaggle 数据下载工具
from rdagent.scenarios.kaggle.kaggle_crawler import download_data


class DataScienceRDLoop(RDLoop):
    """数据科学研发循环类，继承自基础研发循环类"""
    # 定义可跳过的循环错误类型
    skip_loop_error = (CoderError, RunnerError)

    def __init__(self, PROP_SETTING: BasePropSetting):
        """初始化数据科学研发循环
        Args:
            PROP_SETTING: 基础属性设置
        """
        # 记录竞赛信息
        logger.log_object(PROP_SETTING.competition, tag="competition")
        # 导入并初始化场景
        scen: Scenario = import_class(PROP_SETTING.scen)(PROP_SETTING.competition)

        # 初始化知识库（如果配置了的话）
        knowledge_base = (
            import_class(PROP_SETTING.knowledge_base)(PROP_SETTING.knowledge_base_path, scen)
            if PROP_SETTING.knowledge_base != ""
            else None
        )

        # 初始化实验生成器
        self.exp_gen = DSExpGen(scen)
        # 初始化各个组件的代码生成器
        self.data_loader_coder = DataLoaderCoSTEER(scen)
        self.feature_coder = FeatureCoSTEER(scen)
        self.model_coder = ModelCoSTEER(scen)
        self.ensemble_coder = EnsembleCoSTEER(scen)
        self.workflow_coder = WorkflowCoSTEER(scen)

        # 初始化运行器
        self.runner = DSCoSTEERRunner(scen)
        
        # 初始化追踪器和总结器
        self.trace = DSTrace(scen=scen)
        self.summarizer = DSExperiment2Feedback(scen)
        # 调用父类初始化
        super(RDLoop, self).__init__()

    def direct_exp_gen(self, prev_out: dict[str, Any]):
        """直接生成实验
        Args:
            prev_out: 前一步骤的输出
        Returns:
            生成的实验
        """
        # 根据追踪记录生成新实验
        exp = self.exp_gen.gen(self.trace)
        logger.log_object(exp)
        logger.log_object(exp, tag="debug_exp_gen")
        return exp

    def coding(self, prev_out: dict[str, Any]):
        """代码生成阶段
        Args:
            prev_out: 前一步骤的输出
        Returns:
            处理后的实验
        """
        # 获取实验对象
        exp = prev_out["direct_exp_gen"]
        # 处理每个待处理任务列表
        for tasks in exp.pending_tasks_list:
            exp.sub_tasks = tasks
            # 根据任务类型选择相应的代码生成器
            if isinstance(exp.sub_tasks[0], DataLoaderTask):
                exp = self.data_loader_coder.develop(exp)
            elif isinstance(exp.sub_tasks[0], FeatureTask):
                exp = self.feature_coder.develop(exp)
            elif isinstance(exp.sub_tasks[0], ModelTask):
                exp = self.model_coder.develop(exp)
            elif isinstance(exp.sub_tasks[0], EnsembleTask):
                exp = self.ensemble_coder.develop(exp)
            elif isinstance(exp.sub_tasks[0], WorkflowTask):
                exp = self.workflow_coder.develop(exp)
            else:
                raise NotImplementedError(f"Unsupported component in DataScienceRDLoop: {exp.hypothesis.component}")
            exp.sub_tasks = []
        logger.log_object(exp)
        return exp

    def running(self, prev_out: dict[str, Any]):
        """运行阶段
        Args:
            prev_out: 前一步骤的输出
        Returns:
            运行后的实验
        """
        exp: DSExperiment = prev_out["coding"]
        # 如果实验准备就绪则运行
        if exp.is_ready_to_run():
            new_exp = self.runner.develop(exp)
            logger.log_object(new_exp)
            return new_exp
        return exp

    def feedback(self, prev_out: dict[str, Any]) -> ExperimentFeedback:
        """反馈阶段
        Args:
            prev_out: 前一步骤的输出
        Returns:
            实验反馈
        """
        exp: DSExperiment = prev_out["running"]
        # 如果所有组件都已完成，生成完整反馈
        if self.trace.next_incomplete_component() is None:
            feedback = self.summarizer.generate_feedback(exp, self.trace)
        else:
            # 否则生成简单反馈
            feedback = ExperimentFeedback(
                reason=f"{exp.hypothesis.component} is completed.",
                decision=True,
            )
        logger.log_object(feedback)
        return feedback

    def record(self, prev_out: dict[str, Any]):
        """记录阶段
        Args:
            prev_out: 前一步骤的输出
        """
        # 检查是否有异常
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is None:
            # 正常情况：记录运行结果和反馈
            self.trace.hist.append((prev_out["running"], prev_out["feedback"]))
        else:
            # 异常情况：记录异常信息
            self.trace.hist.append(
                (
                    prev_out["direct_exp_gen"] if isinstance(e, CoderError) else prev_out["coding"],
                    ExperimentFeedback.from_exception(e),
                )
            )
            # 检查是否需要重启
            if self.trace.sota_experiment() is None and len(self.trace.hist) >= DS_RD_SETTING.consecutive_errors:
                for _, fb in self.trace.hist[-DS_RD_SETTING.consecutive_errors :]:
                    if fb:
                        break
                else:
                    logger.error("Consecutive errors reached the limit. Dumping trace.")
                    logger.log_object(self.trace, tag="trace before restart")
                    self.trace = DSTrace(scen=self.trace.scen, knowledge_base=self.trace.knowledge_base)
        # 记录追踪信息
        logger.log_object(self.trace, tag="trace")
        logger.log_object(self.trace.sota_experiment(), tag="SOTA experiment")


def main(path=None, output_path=None, step_n=None, loop_n=None, competition="bms-molecular-translation"):
    """主函数
    Args:
        path: 恢复状态的路径
        output_path: 输出路径
        step_n: 运行步数
        loop_n: 循环次数
        competition: 竞赛名称
    """
    # 设置竞赛
    if competition is not None:
        DS_RD_SETTING.competition = competition

    # 处理竞赛数据
    if DS_RD_SETTING.competition:
        if DS_RD_SETTING.scen.endswith("KaggleScen"):
            # Kaggle 场景：下载数据
            download_data(competition=DS_RD_SETTING.competition, settings=DS_RD_SETTING)
        else:
            # 其他场景：检查数据是否存在
            if not Path(f"{DS_RD_SETTING.local_data_path}/{competition}").exists():
                logger.error(f"Please prepare data for competition {competition} first.")
                return
    else:
        logger.error("Please specify competition name.")
        
    # 创建或加载研发循环实例
    if path is None:
        kaggle_loop = DataScienceRDLoop(DS_RD_SETTING)
    else:
        kaggle_loop = DataScienceRDLoop.load(path, output_path)
    # 运行研发循环
    kaggle_loop.run(step_n=step_n, loop_n=loop_n)


# 程序入口
if __name__ == "__main__":
    fire.Fire(main)
