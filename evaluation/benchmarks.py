"""多数据集、多模型的 Benchmark 运行器。

对比方案：
  - Qwen3-VL zero-shot（无任何微调）
  - GPT-4V / GPT-4o zero-shot（API 调用）
  - BLIP-2 + LoRA 微调
  - Ours：Qwen3-VL + LoRA 微调

生成用于论文/README 的结构化对比表格。

用法：
    >>> runner = BenchmarkRunner(output_dir="experiments/benchmarks")
    >>> runner.add_baseline("Zero-shot Qwen3-VL", "rsicd", bleu_4=12.3, ...)
    >>> runner.load_ours("output/eval_results_300.json", method="Ours (r=8)")
    >>> runner.print_table()
    >>> runner.save_report()   # 生成 Markdown + LaTeX
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 已知的 RSICD 文献方法（供参考对比）──
KNOWN_BASELINES = {
    "SAT": {"bleu_4": 28.0, "meteor": 26.8, "rouge_l": 49.5, "cider_d": 87.0, "spice": 23.5},
    "Adaptive": {"bleu_4": 30.1, "meteor": 28.3, "rouge_l": 53.6, "cider_d": 113.9, "spice": 29.0},
    "Up-Down": {"bleu_4": 31.3, "meteor": 29.1, "rouge_l": 54.5, "cider_d": 124.8, "spice": 31.1},
    "GCN-LSTM": {"bleu_4": 33.4, "meteor": 30.0, "rouge_l": 56.5, "cider_d": 138.6, "spice": 33.2},
    "Transformer": {"bleu_4": 35.7, "meteor": 31.5, "rouge_l": 58.2, "cider_d": 155.1, "spice": 36.5},
    "Ours (Qwen3-VL-2B + LoRA r=8)": None,  # 将由实际评估填充
}


@dataclass
class BenchmarkResult:
    """单次 benchmark 运行的结构化结果。"""
    method: str               # 方法名称
    dataset: str              # 数据集名称
    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_3: float = 0.0
    bleu_4: float = 0.0
    meteor: float = 0.0
    rouge_l: float = 0.0
    cider_d: float = 0.0
    spice: float = 0.0
    hallucination_rate: Optional[float] = None   # 幻觉率（可选）
    n_samples: int = 0
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


class BenchmarkRunner:
    """编排多模型、多数据集的评估与对比流程。

    使用方式：
        runner = BenchmarkRunner()
        # 方式 1：手动添加基线
        runner.add_baseline("Zero-shot Qwen3-VL", "rsicd", bleu_4=8.5, ...)
        # 方式 2：从评估结果文件加载
        runner.load_ours("output/eval_results_300.json", method="Ours (r=8)")
        # 生成报告
        runner.print_table()
        runner.save_report()
    """

    def __init__(self, output_dir: str = "experiments/benchmarks"):
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []
        os.makedirs(output_dir, exist_ok=True)

    # ════════════════════════════════════════════════════════════
    # 数据录入
    # ════════════════════════════════════════════════════════════

    def add_baseline(
        self,
        method: str,
        dataset: str,
        bleu_1: float = 0.0,
        bleu_2: float = 0.0,
        bleu_3: float = 0.0,
        bleu_4: float = 0.0,
        meteor: float = 0.0,
        rouge_l: float = 0.0,
        cider_d: float = 0.0,
        spice: float = 0.0,
        hallucination_rate: Optional[float] = None,
        n_samples: int = 0,
        extra_metrics: Optional[Dict[str, float]] = None,
        notes: str = "",
    ):
        """手动添加一个 baseline 结果。

        适用于：文献中的结果、GPT-4V API 结果等无法自动运行的 baseline。
        """
        result = BenchmarkResult(
            method=method,
            dataset=dataset.lower(),
            bleu_1=bleu_1,
            bleu_2=bleu_2,
            bleu_3=bleu_3,
            bleu_4=bleu_4,
            meteor=meteor,
            rouge_l=rouge_l,
            cider_d=cider_d,
            spice=spice,
            hallucination_rate=hallucination_rate,
            n_samples=n_samples,
            extra_metrics=extra_metrics or {},
            notes=notes,
        )
        self.results.append(result)
        logger.info(f"已添加 baseline: {method} @ {dataset}")

    def load_ours(
        self,
        eval_result_path: str,
        method: str = "Ours",
        dataset: Optional[str] = None,
    ):
        """从评估结果 JSON 文件加载我们的模型结果。

        参数：
            eval_result_path: eval.py 或 eval_rsicd.py 输出的 JSON 文件路径。
            method: 方法名称（如 "Ours (LoRA r=8)"）。
            dataset: 数据集名称。为 None 时从 JSON 推断。
        """
        with open(eval_result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if dataset is None:
            dataset = data.get("dataset", "rsicd").lower()

        metrics_raw = data.get("metrics", {})
        # 兼容两种 JSON 格式：
        #   旧格式：{"metrics": {"bleu_4": 0.212, ...}}
        #   新格式：{"metrics": {"global": {"bleu_4": 0.212, ...}, "per_category": {...}}}
        if "global" in metrics_raw:
            metrics = metrics_raw["global"]
        else:
            metrics = metrics_raw

        n_samples = data.get("n_samples", len(data.get("samples", [])))

        # pycocoevalcap 的 BLEU/CIDEr 返回 [0,1] 分数，
        # ROUGE 返回 [0,1]，文献中通常 ×100 报告百分比。
        # 此处统一转为百分比（×100），方便与文献对比。
        _scale = 100.0

        result = BenchmarkResult(
            method=method,
            dataset=dataset,
            bleu_1=metrics.get("bleu_1", 0.0) * _scale,
            bleu_2=metrics.get("bleu_2", 0.0) * _scale,
            bleu_3=metrics.get("bleu_3", 0.0) * _scale,
            bleu_4=metrics.get("bleu_4", 0.0) * _scale,
            meteor=metrics.get("meteor", 0.0) * _scale,
            rouge_l=metrics.get("rouge_l", 0.0) * _scale,
            cider_d=metrics.get("cider_d", 0.0),
            spice=metrics.get("spice", 0.0),
            n_samples=n_samples,
            notes=f"从 {eval_result_path} 加载",
        )
        self.results.append(result)
        logger.info(f"已加载 ours: {method} @ {dataset} ({n_samples} samples)")

    def load_error_analysis(
        self,
        error_analysis_path: str,
        method: str = "Ours",
        dataset: Optional[str] = None,
    ):
        """从错误分析 JSON 补充幻觉率等额外指标。"""
        with open(error_analysis_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if dataset is None:
            dataset = data.get("dataset", "unknown")

        summary = data.get("summary", {})
        hallucination_rate = summary.get("hallucination_rate", None)

        # 查找已有结果并更新
        for r in self.results:
            if r.method == method and r.dataset == dataset:
                r.hallucination_rate = hallucination_rate
                logger.info(f"已更新 {method} @ {dataset} 的幻觉率: {hallucination_rate}")
                return

        # 如果不存在，创建一个仅含幻觉率的结果
        result = BenchmarkResult(
            method=method,
            dataset=dataset,
            hallucination_rate=hallucination_rate,
            notes=f"从 {error_analysis_path} 加载",
        )
        self.results.append(result)

    def add_known_baselines(self, dataset: str = "rsicd"):
        """批量添加文献中已知的 baseline 方法。"""
        dataset = dataset.lower()
        for method, metrics in KNOWN_BASELINES.items():
            if metrics is None:
                continue
            self.add_baseline(
                method=method,
                dataset=dataset,
                bleu_4=metrics.get("bleu_4", 0.0),
                meteor=metrics.get("meteor", 0.0),
                rouge_l=metrics.get("rouge_l", 0.0),
                cider_d=metrics.get("cider_d", 0.0),
                spice=metrics.get("spice", 0.0),
                notes="文献结果 (取自 RSICD benchmark)",
            )
        logger.info(f"已添加 {len(KNOWN_BASELINES) - 1} 个已知 baseline")

    # ════════════════════════════════════════════════════════════
    # 报告生成
    # ════════════════════════════════════════════════════════════

    def _get_methods(self) -> List[str]:
        """获取所有唯一的方法名（保持插入顺序）。"""
        seen = []
        for r in self.results:
            if r.method not in seen:
                seen.append(r.method)
        return seen

    def _get_datasets(self) -> List[str]:
        """获取所有唯一的数据集名。"""
        seen = []
        for r in self.results:
            if r.dataset not in seen:
                seen.append(r.dataset)
        return seen

    def _get_result(self, method: str, dataset: str) -> Optional[BenchmarkResult]:
        """查找指定方法和数据集的评估结果。"""
        for r in self.results:
            if r.method == method and r.dataset == dataset:
                return r
        return None

    def print_table(
        self,
        metrics: Optional[List[str]] = None,
        sort_by: str = "bleu_4",
    ):
        """打印格式化的对比表格到控制台。

        参数：
            metrics: 要显示的指标列表。默认：bleu_4, meteor, rouge_l, cider_d, spice。
            sort_by: 按哪个指标排序。默认 BLEU-4。
        """
        if metrics is None:
            metrics = ["bleu_4", "meteor", "rouge_l", "cider_d", "spice"]

        datasets = self._get_datasets()
        methods = self._get_methods()

        for dataset in datasets:
            print(f"\n{'=' * 70}")
            print(f"  Dataset: {dataset.upper()}")
            print(f"{'=' * 70}")

            # ── 表头 ──────────────────────
            header = f"  {'Method':30s}"
            for m in metrics:
                header += f" {m:>8s}"
            print(header)
            print(f"  {'-' * (30 + 9 * len(metrics))}")

            # ── 收集并按 sort_by 排序 ──────
            dataset_results = []
            for method in methods:
                r = self._get_result(method, dataset)
                if r:
                    dataset_results.append((method, r))

            # 按指定指标降序
            dataset_results.sort(
                key=lambda x: getattr(x[1], sort_by, 0.0), reverse=True
            )

            # ── 逐行打印 ──────────────────
            for method, r in dataset_results:
                row = f"  {method:30s}"
                for m in metrics:
                    val = getattr(r, m, 0.0)
                    row += f" {val:8.1f}"
                print(row)

    def generate_markdown_table(
        self,
        metrics: Optional[List[str]] = None,
        sort_by: str = "bleu_4",
    ) -> str:
        """生成 Markdown 格式的对比表格。

        返回：
            Markdown 字符串，包含所有数据集的对比表。
        """
        if metrics is None:
            metrics = ["bleu_4", "meteor", "rouge_l", "cider_d", "spice"]

        datasets = self._get_datasets()
        methods = self._get_methods()

        lines = []
        lines.append("# Qwen-VL-RS Benchmark 结果\n")
        lines.append(f"> 生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        for dataset in datasets:
            lines.append(f"## {dataset.upper()}\n")

            # 表头
            header_cols = ["方法"] + [m.upper().replace("_", "-") for m in metrics]
            lines.append("| " + " | ".join(header_cols) + " |")
            lines.append("|" + "---|" * len(header_cols))

            # 收集并排序
            dataset_results = []
            for method in methods:
                r = self._get_result(method, dataset)
                if r:
                    dataset_results.append((method, r))

            dataset_results.sort(
                key=lambda x: getattr(x[1], sort_by, 0.0), reverse=True
            )

            # 标记 ours
            ours_methods = {m for m in methods if m.lower().startswith("ours")}

            for method, r in dataset_results:
                suffix = " **(ours)**" if method in ours_methods else ""
                cols = [f"{method}{suffix}"]
                for m in metrics:
                    val = getattr(r, m, 0.0)
                    cols.append(f"{val:.1f}")
                lines.append("| " + " | ".join(cols) + " |")

            lines.append("")

        return "\n".join(lines)

    def generate_latex_table(
        self,
        metrics: Optional[List[str]] = None,
        sort_by: str = "bleu_4",
    ) -> str:
        """生成 LaTeX 格式的对比表格（用于论文）。

        返回：
            LaTeX 表格代码。
        """
        if metrics is None:
            metrics = ["bleu_4", "meteor", "rouge_l", "cider_d", "spice"]

        datasets = self._get_datasets()
        methods = self._get_methods()
        ours_methods = {m for m in methods if m.lower().startswith("ours")}

        lines = []
        lines.append("% Qwen-VL-RS Benchmark — 自动生成")
        lines.append("% 日期: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        lines.append("")

        for dataset in datasets:
            col_spec = "l" + "c" * len(metrics)
            lines.append(f"\\begin{{table}}[htbp]")
            lines.append(f"  \\centering")
            lines.append(
                f"  \\caption{{{dataset.upper()} 数据集上的评估结果对比}}"
            )
            lines.append(f"  \\label{{tab:benchmark_{dataset}}}")

            header_str = " & ".join(
                ["方法"] + [m.upper().replace("_", "-") for m in metrics]
            )
            lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
            lines.append(f"    \\toprule")
            lines.append(f"    {header_str} \\\\")
            lines.append(f"    \\midrule")

            # 收集并排序
            dataset_results = []
            for method in methods:
                r = self._get_result(method, dataset)
                if r:
                    dataset_results.append((method, r))

            dataset_results.sort(
                key=lambda x: getattr(x[1], sort_by, 0.0), reverse=True
            )

            for method, r in dataset_results:
                is_ours = method in ours_methods
                prefix = "    \\textbf{" if is_ours else "    "
                suffix = "}" if is_ours else ""
                cols = [f"{prefix}{method}{suffix}"]
                for m in metrics:
                    val = getattr(r, m, 0.0)
                    cols.append(f"{prefix}{val:.1f}{suffix}")
                lines.append(" & ".join(cols) + " \\\\")

            lines.append(f"    \\bottomrule")
            lines.append(f"  \\end{{tabular}}")
            lines.append(f"\\end{{table}}")
            lines.append("")

        return "\n".join(lines)

    def save_report(
        self,
        metrics: Optional[List[str]] = None,
        filename_prefix: str = "benchmark_report",
    ):
        """保存 Markdown 和 LaTeX 报告到文件。"""
        markdown = self.generate_markdown_table(metrics)
        latex = self.generate_latex_table(metrics)

        md_path = os.path.join(self.output_dir, f"{filename_prefix}.md")
        tex_path = os.path.join(self.output_dir, f"{filename_prefix}.tex")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex)

        logger.info(f"Markdown 报告: {md_path}")
        logger.info(f"LaTeX 报告:   {tex_path}")

        return md_path, tex_path

    def save_results_json(self, filename: str = "benchmark_results.json"):
        """将所有 benchmark 结果导出为 JSON。"""
        output = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": [asdict(r) for r in self.results],
        }
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"结果 JSON: {path}")
        return path


# ════════════════════════════════════════════════════════════════
# CLI 入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen-VL-RS Benchmark 运行器")
    parser.add_argument("--output_dir", type=str, default="experiments/benchmarks",
                        help="报告输出目录")
    parser.add_argument("--ours", type=str, nargs="*", default=[],
                        help="我们的评估结果 JSON 文件路径（可多个）")
    parser.add_argument("--ours_method", type=str, default="Ours (LoRA r=8)",
                        help="我们的方法名称")
    parser.add_argument("--error_analysis", type=str, nargs="*", default=[],
                        help="错误分析 JSON 文件路径（补充幻觉率）")
    parser.add_argument("--add_known_baselines", action="store_true",
                        help="添加文献中已知的 baseline 方法")
    parser.add_argument("--baseline_file", type=str, default=None,
                        help="手动 baseline 的 JSON 文件路径")

    args = parser.parse_args()

    runner = BenchmarkRunner(output_dir=args.output_dir)

    # 加载我们的结果
    for eval_path in args.ours:
        if os.path.exists(eval_path):
            runner.load_ours(eval_path, method=args.ours_method)
        else:
            logger.warning(f"文件不存在，跳过: {eval_path}")

    # 补充错误分析
    for ea_path in args.error_analysis:
        if os.path.exists(ea_path):
            runner.load_error_analysis(ea_path, method=args.ours_method)

    # 添加已知 baseline
    if args.add_known_baselines:
        runner.add_known_baselines()

    # 加载手动 baseline
    if args.baseline_file and os.path.exists(args.baseline_file):
        with open(args.baseline_file, "r") as f:
            baselines = json.load(f)
            for bl in baselines:
                runner.add_baseline(**bl)

    # 输出
    runner.print_table()
    runner.save_report()

    if runner.results:
        runner.save_results_json()
