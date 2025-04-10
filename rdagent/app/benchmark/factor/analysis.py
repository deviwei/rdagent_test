# 导入 JSON 处理库
import json
# 导入序列化库
import pickle
# 导入路径处理库
from pathlib import Path

# 导入命令行参数处理库
import fire
# 导入绘图库
import matplotlib.pyplot as plt
# 导入数值计算库
import numpy as np
# 导入数据处理库
import pandas as pd
# 导入统计可视化库
import seaborn as sns

# 导入基准测试配置
from rdagent.components.benchmark.conf import BenchmarkSettings
# 导入因子实现评估工具
from rdagent.components.benchmark.eval_method import FactorImplementEval


class BenchmarkAnalyzer:
    """基准测试分析器，用于分析和处理因子实现的评估结果"""
    
    def __init__(self, settings, only_correct_format=False):
        """初始化分析器"""
        # 保存基准测试设置
        self.settings = settings
        # 加载因子索引映射
        self.index_map = self.load_index_map()
        # 是否只分析格式正确的结果
        self.only_correct_format = only_correct_format

    def load_index_map(self):
        """从基准数据文件中加载因子索引映射"""
        # 创建空字典存储映射关系
        index_map = {}
        # 打开并读取基准数据文件
        with open(self.settings.bench_data_path, "r") as file:
            factor_dict = json.load(file)
        # 遍历因子字典，构建映射关系
        for factor_name, data in factor_dict.items():
            index_map[factor_name] = (factor_name, data["Category"], data["Difficulty"])
        return index_map

    def load_data(self, file_path):
        """加载评估结果数据"""
        # 转换为 Path 对象
        file_path = Path(file_path)
        # 验证文件路径和后缀
        if not (file_path.is_file() and file_path.suffix == ".pkl"):
            raise ValueError("Invalid file path")
        # 读取 pickle 文件
        with file_path.open("rb") as f:
            res = pickle.load(f)
        return res

    def process_results(self, results):
        """处理多个实验结果"""
        # 创建空字典存储处理结果
        final_res = {}
        # 遍历实验结果
        for experiment, path in results.items():
            # 加载数据
            data = self.load_data(path)
            # 汇总结果
            summarized_data = FactorImplementEval.summarize_res(data)
            # 分析数据
            processed_data = self.analyze_data(summarized_data)
            # 存储最终行结果
            final_res[experiment] = processed_data.iloc[-1, :]
        return final_res

    def reformat_index(self, display_df):
        """
        重新格式化结果索引，将单层索引转换为多层索引
        包含类别（Category）、难度（Difficulty）和因子（Factor）
        """
        new_idx = []
        display_df = display_df[display_df.index.isin(self.index_map.keys())]
        for idx in display_df.index:
            new_idx.append(self.index_map[idx])

        display_df.index = pd.MultiIndex.from_tuples(
            new_idx,
            names=["Factor", "Category", "Difficulty"],
        )
        display_df = display_df.swaplevel(0, 2).swaplevel(0, 1).sort_index(axis=0)

        return display_df.sort_index(
            key=lambda x: [{"Easy": 0, "Medium": 1, "Hard": 2, "New Discovery": 3}.get(i, i) for i in x]
        )

    def result_all_key_order(self, x):
        order_v = []
        for i in x:
            order_v.append(
                {
                    "Avg Run SR": 0,
                    "Avg Format SR": 1,
                    "Avg Correlation": 2,
                    "Max Correlation": 3,
                    "Max Accuracy": 4,
                    "Avg Accuracy": 5,
                }.get(i, i),
            )
        return order_v

    def analyze_data(self, sum_df):
        """
        分析评估数据，计算各种指标
        包括运行成功率、格式正确率、相关性等
        @return: 包含所有分析结果的DataFrame
        """
        index = [
            "FactorSingleColumnEvaluator",
            "FactorRowCountEvaluator",
            "FactorIndexEvaluator",
            "FactorEqualValueRatioEvaluator",
            "FactorCorrelationEvaluator",
            "run factor error",
        ]
        sum_df = sum_df.reindex(index, axis=0)
        sum_df_clean = sum_df.T.groupby(level=0).apply(lambda x: x.reset_index(drop=True))

        run_error = sum_df_clean["run factor error"].unstack().T.fillna(False).astype(bool)
        succ_rate = ~run_error
        succ_rate = succ_rate.mean(axis=0).to_frame("success rate")

        succ_rate_f = self.reformat_index(succ_rate)

        # if it rasis Error when running the evaluator, we will get NaN
        # Running failures are reguarded to zero score.
        format_issue = sum_df_clean[["FactorRowCountEvaluator", "FactorIndexEvaluator"]].apply(
            lambda x: np.mean(x.fillna(0.0)), axis=1
        )
        format_succ_rate = format_issue.unstack().T.mean(axis=0).to_frame("success rate")
        format_succ_rate_f = self.reformat_index(format_succ_rate)

        corr = sum_df_clean["FactorCorrelationEvaluator"].fillna(0.0)
        if self.only_correct_format:
            corr = corr.loc[format_issue == 1.0]

        corr_res = corr.unstack().T.mean(axis=0).to_frame("corr(only success)")
        corr_res = self.reformat_index(corr_res)

        corr_max = corr.unstack().T.max(axis=0).to_frame("corr(only success)")
        corr_max_res = self.reformat_index(corr_max)

        value_max = sum_df_clean["FactorEqualValueRatioEvaluator"]
        value_max = value_max.unstack().T.max(axis=0).to_frame("max_value")
        value_max_res = self.reformat_index(value_max)

        value_avg = (
            (sum_df_clean["FactorEqualValueRatioEvaluator"] * format_issue)
            .unstack()
            .T.mean(axis=0)
            .to_frame("avg_value")
        )
        value_avg_res = self.reformat_index(value_avg)

        result_all = pd.concat(
            {
                "Avg Correlation": corr_res.iloc[:, 0],
                "Avg Format SR": format_succ_rate_f.iloc[:, 0],
                "Avg Run SR": succ_rate_f.iloc[:, 0],
                "Max Correlation": corr_max_res.iloc[:, 0],
                "Max Accuracy": value_max_res.iloc[:, 0],
                "Avg Accuracy": value_avg_res.iloc[:, 0],
            },
            axis=1,
        )

        df = result_all.sort_index(axis=1, key=self.result_all_key_order).sort_index(axis=0)
        print(df)

        print()
        print(df.groupby("Category").mean())

        print()
        print(df.mean())

        # Calculate the mean of each column
        mean_values = df.fillna(0.0).mean()
        mean_df = pd.DataFrame(mean_values).T

        # Assign the MultiIndex to the DataFrame
        mean_df.index = pd.MultiIndex.from_tuples([("-", "-", "Average")], names=["Factor", "Category", "Difficulty"])

        # Append the mean values to the end of the dataframe
        df_w_mean = pd.concat([df, mean_df]).astype("float")

        return df_w_mean


class Plotter:
    """结果可视化工具类"""
    
    @staticmethod
    def change_fs(font_size):
        """设置图表字体大小"""
        plt.rc("font", size=font_size)
        plt.rc("axes", titlesize=font_size)
        plt.rc("axes", labelsize=font_size)
        plt.rc("xtick", labelsize=font_size)
        plt.rc("ytick", labelsize=font_size)
        plt.rc("legend", fontsize=font_size)
        plt.rc("figure", titlesize=font_size)

    @staticmethod
    def plot_data(data, file_name, title):
        """
        绘制条形图并保存
        @param data: 要绘制的数据
        @param file_name: 输出文件名
        @param title: 图表标题
        """
        plt.figure(figsize=(10, 10))
        plt.ylabel("Value")
        colors = ["#3274A1", "#E1812C", "#3A923A", "#C03D3E"]
        plt.bar(data["a"], data["b"], color=colors, capsize=5)
        for idx, row in data.iterrows():
            plt.text(idx, row["b"] + 0.01, f"{row['b']:.2f}", ha="center", va="bottom")
        plt.suptitle(title, y=0.98)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(file_name)


def main(
    path="git_ignore_folder/eval_results/res_promptV220240724-060037.pkl",
    round=1,
    title="Comparison of Different Methods",
    only_correct_format=False,
):
    """
    主函数：运行基准测试分析
    @param path: 评估结果文件路径
    @param round: 实验轮次
    @param title: 图表标题
    @param only_correct_format: 是否只分析格式正确的结果
    """
    settings = BenchmarkSettings()
    benchmark = BenchmarkAnalyzer(settings, only_correct_format=only_correct_format)
    results = {
        f"{round} round experiment": path,
    }
    final_results = benchmark.process_results(results)
    final_results_df = pd.DataFrame(final_results)

    Plotter.change_fs(20)
    plot_data = final_results_df.drop(["Max Accuracy", "Avg Accuracy"], axis=0).T
    plot_data = plot_data.reset_index().melt("index", var_name="a", value_name="b")
    Plotter.plot_data(plot_data, "./comparison_plot.png", title)


if __name__ == "__main__":
    fire.Fire(main)
