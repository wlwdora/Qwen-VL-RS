"""多数据集、多模型的 Benchmark 运行器。

对比方案：
  - Qwen3-VL zero-shot（无任何微调）
  - GPT-4V / GPT-4o zero-shot（API 调用）
  - BLIP-2 + LoRA 微调
  - Ours：Qwen3-VL + LoRA 微调

生成用于论文/README 的结构化对比表格。
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BenchmarkResult:
    """单次 benchmark 运行的结构化结果。"""
    method: str               # 方法名称
    dataset: str              # 数据集名称
    bleu_4: float = 0.0
    meteor: float = 0.0
    rouge_l: float = 0.0
    cider_d: float = 0.0
    spice: float = 0.0
    extra_metrics: Dict[str, float] = field(default_factory=dict)


class BenchmarkRunner:
    """编排多模型、多数据集的评估流程。"""

    def __init__(self, output_dir: str = "experiments/benchmarks"):
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []

    def run_all(self):
        """运行所有配置的 benchmark。"""
        # TODO: 实现 benchmark 运行器
        # 1. 运行 zero-shot baseline
        # 2. 运行 BLIP-2 baseline
        # 3. 在所有数据集上评估我们的模型
        # 4. 如果有 API key，运行 GPT-4V 评估
        # 5. 生成对比表格
        raise NotImplementedError("【待实现】Benchmark")

    def generate_report(self) -> str:
        """生成 Markdown 格式的对比报告。"""
        # TODO: 实现报告生成
        raise NotImplementedError("【待实现】报告生成")
