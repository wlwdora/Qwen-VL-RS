# 遥感图像描述 —— 评估模块
# 多数据集 benchmark、错误分析、与 baseline 对比

from .eval import CaptionEvaluator
from .benchmarks import BenchmarkRunner
from .error_analysis import ErrorAnalyzer

__all__ = [
    "CaptionEvaluator",
    "BenchmarkRunner",
    "ErrorAnalyzer",
]
