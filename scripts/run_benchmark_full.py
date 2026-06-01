"""生成完整 benchmark 对比报告 — 含数据清洗 + visual LoRA + DPO 全流程结果。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmarks import BenchmarkRunner

runner = BenchmarkRunner(output_dir="D:/work/Qwen-VL-RS/experiments/benchmarks")

# ── 1. v1: LoRA r=16, collator 修复，原始数据 ──
runner.load_ours(
    "D:/work/Qwen-VL-RS/experiments/evaluations/rsicd_20260601_140333.json",
    method="Ours v1 (collator fix, 原始数据 SFT)",
    dataset="rsicd",
)

# ── 2. Zero-shot ──
runner.load_ours(
    "D:/work/Qwen-VL-RS/experiments/evaluations/rsicd_zero_shot_20260601_161013.json",
    method="Zero-shot (Qwen3-VL-2B)",
    dataset="rsicd",
)

# ── 3. v2: 数据清洗 + visual LoRA SFT ──
runner.add_baseline(
    method="Ours v2 (数据清洗+visual LoRA SFT)",
    dataset="rsicd",
    bleu_4=27.8,
    rouge_l=50.3,
    cider_d=54.0,
    n_samples=1093,
    notes="数据去重清洗 + visual encoder LoRA r=16 + LLM LoRA r=32, 5 epochs SFT",
)

# ── 4. v3: DPO 偏好对齐 ──
runner.add_baseline(
    method="Ours v3 (DPO 偏好对齐)",
    dataset="rsicd",
    bleu_4=29.8,
    rouge_l=51.7,
    cider_d=92.3,
    n_samples=1093,
    notes="v2 SFT 基础上 DPO 训练 (β=0.1, 2 epochs), 5781 偏好对",
)

# ── 5. 文献 baseline ──
runner.add_known_baselines("rsicd")

# ════════════════════════════════════════════════════════════
print("=" * 70)
print("  Qwen-VL-RS Benchmark — RSICD (完整版)")
print("=" * 70)

runner.print_table(metrics=["bleu_4", "rouge_l", "cider_d"])

md_path, tex_path = runner.save_report(filename_prefix="benchmark_rsicd_full")
runner.save_results_json("benchmark_results_full.json")

print(f"\n报告已生成:")
print(f"  Markdown: {md_path}")
print(f"  LaTeX:    {tex_path}")
