"""生成 benchmark 对比报告 — zero-shot vs LoRA r=16 vs 文献方法。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmarks import BenchmarkRunner

runner = BenchmarkRunner(output_dir="D:/work/Qwen-VL-RS/experiments/benchmarks")

# ── 1. 加载我们的 LoRA r=16 结果 ─────────────────
runner.load_ours(
    "D:/work/Qwen-VL-RS/experiments/evaluations/rsicd_20260601_140333.json",
    method="Ours (LoRA r=16, fixed)",
    dataset="rsicd",
)

# ── 2. 加载 zero-shot 结果 ─────────────────────
runner.load_ours(
    "D:/work/Qwen-VL-RS/experiments/evaluations/rsicd_zero_shot_20260601_161013.json",
    method="Zero-shot (Qwen3-VL-2B, no LoRA)",
    dataset="rsicd",
)

# ── 3. 添加文献已知 baseline (RSICD benchmark) ──
runner.add_known_baselines("rsicd")

# ════════════════════════════════════════════════════════════
# 生成报告
# ════════════════════════════════════════════════════════════

print("=" * 70)
print("  Qwen-VL-RS Benchmark — RSICD 数据集")
print("=" * 70)

# 打印控制台表格
runner.print_table(metrics=["bleu_4", "rouge_l", "cider_d"])

# 生成 Markdown + LaTeX
md_path, tex_path = runner.save_report(filename_prefix="benchmark_rsicd")
runner.save_results_json("benchmark_results.json")

print(f"\n报告已生成:")
print(f"  Markdown: {md_path}")
print(f"  LaTeX:    {tex_path}")
