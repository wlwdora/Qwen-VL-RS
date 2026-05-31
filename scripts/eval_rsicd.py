"""RSICD 评估脚本 — 独立入口，避免模块导入 segfault。"""
import sys, os, gc, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
torch.manual_seed(42)
torch.cuda.empty_cache()
gc.collect()

from data.dataset import RemoteSensingCaptionDataset
from data.collator import MultiModalDataCollator
from models.qwen_vl_rs import QwenVLForRemoteSensing

# ── 配置 ────────────────────────────────
MODEL_PATH = "D:/Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_PATH = "D:/work/Qwen-VL-RS/output/qwen_vl_rs_lora/best_model"
DATA_PATH = "D:/work/Qwen-VL-RS/data/processed/rsicd.jsonl"
OUTPUT_DIR = "D:/work/Qwen-VL-RS/output/qwen_vl_rs_lora"
DEVICE = "cuda"

MAX_SAMPLES = 300   # 抽样数，设为 0 表示全部
BATCH_SIZE = 2
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  Qwen-VL-RS 评估 — RSICD Test Set")
print("=" * 60)

# ════════════════════════════════════════════════════════════
# 1. 加载模型
# ════════════════════════════════════════════════════════════
print("\n[1/4] 加载模型...")
wrapper = QwenVLForRemoteSensing.from_pretrained(
    model_path=MODEL_PATH,
    lora_adapter_path=ADAPTER_PATH,
)
model = wrapper.peft_model
model.eval()
print(f"  LoRA adapter 已加载: {ADAPTER_PATH}")

# ════════════════════════════════════════════════════════════
# 2. 加载数据
# ════════════════════════════════════════════════════════════
print("\n[2/4] 加载数据...")
test_ds = RemoteSensingCaptionDataset(
    data_paths=DATA_PATH, split="test", max_length=512,
)
n_total = len(test_ds)
n_eval = min(MAX_SAMPLES, n_total) if MAX_SAMPLES > 0 else n_total
print(f"  Test set: {n_total} samples, evaluating {n_eval} samples")
print(f"  Categories: {len(test_ds.category_counts)}")

# ════════════════════════════════════════════════════════════
# 3. Collator
# ════════════════════════════════════════════════════════════
print("\n[3/4] 构建 Collator...")
collator = MultiModalDataCollator(
    tokenizer=wrapper.tokenizer,
    processor=wrapper.processor,
    max_length=512,
    prompt="请详细描述这张遥感图像的内容。",
)

# ════════════════════════════════════════════════════════════
# 4. 推理 & 收集预测
# ════════════════════════════════════════════════════════════
print("\n[4/4] 推理中...")
predictions = []
references = []
categories = []
image_paths = []

model_device = next(model.parameters()).device

start_time = time.time()
n_batches = 0

for i in range(0, n_eval, BATCH_SIZE):
    end = min(i + BATCH_SIZE, n_eval)
    samples = [test_ds[int(idx)] for idx in range(i, end)]

    try:
        batch = collator(samples)
    except Exception as e:
        print(f"  [WARN] Collator error at sample {i}: {e}")
        continue

    inputs = {k: v.to(model_device) for k, v in batch.items()}

    with torch.no_grad():
        try:
            output_ids = wrapper.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=(TEMPERATURE > 0),
            )
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[:, prompt_len:]
            captions = wrapper.decode(new_tokens, skip_special_tokens=True)
        except Exception as ex:
            print(f"  [WARN] Generate error at sample {i}: {ex}")
            continue

    for j, sample in enumerate(samples):
        predictions.append(captions[j].strip() if j < len(captions) else "")
        references.append(sample["captions"])
        categories.append(sample["category"])
        image_paths.append(sample["image_path"])

    n_batches += 1
    if n_batches % 50 == 0:
        elapsed = time.time() - start_time
        speed = (n_batches * BATCH_SIZE) / elapsed
        eta = (n_eval / BATCH_SIZE - n_batches) * elapsed / max(n_batches, 1)
        print(f"  [{n_batches * BATCH_SIZE}/{n_eval}] {speed:.1f} samples/s, ETA {eta:.0f}s")

elapsed_total = time.time() - start_time
print(f"\n  推理完成! {len(predictions)} predictions in {elapsed_total:.1f}s ({len(predictions)/elapsed_total:.1f} samples/s)")

# ════════════════════════════════════════════════════════════
# 5. 计算指标
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  计算评估指标...")
print("=" * 60)

try:
    from training.metrics import CaptioningMetrics
    metrics_calc = CaptioningMetrics(metrics=["bleu", "meteor", "rouge", "cider", "spice"])
    results = metrics_calc.compute_all_detailed(predictions, references, categories)
    global_metrics = results["global"]
    per_category = results.get("per_category", {})

    print("\n" + metrics_calc.format_results(global_metrics))

    # Per-category top/bottom
    if per_category:
        sorted_cats = sorted(per_category.items(), key=lambda x: x[1].get("bleu_4", 0), reverse=True)
        print(f"\n  Best 5 categories (by BLEU-4):")
        for cat, scores in sorted_cats[:5]:
            print(f"    {cat:25s}: BLEU-4={scores.get('bleu_4',0):.1f}, CIDEr={scores.get('cider_d',0):.1f}")
        print(f"\n  Worst 5 categories (by BLEU-4):")
        for cat, scores in sorted_cats[-5:]:
            print(f"    {cat:25s}: BLEU-4={scores.get('bleu_4',0):.1f}, CIDEr={scores.get('cider_d',0):.1f}")

except ImportError as e:
    print(f"  [WARN] Metrics import failed: {e}, using fallback BLEU")
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    refs_for_bleu = [[r] for r in references]
    bleu_4 = corpus_bleu(refs_for_bleu, predictions, smoothing_function=SmoothingFunction().method3)
    global_metrics = {"bleu_4": bleu_4 * 100}
    print(f"\n  BLEU-4 (corpus): {bleu_4*100:.1f}")

# ════════════════════════════════════════════════════════════
# 6. 保存结果
# ════════════════════════════════════════════════════════════
print("\n保存结果...")
eval_result = {
    "model": "Qwen3-VL-2B-Instruct + LoRA (r=8)",
    "dataset": "RSICD",
    "n_samples": len(predictions),
    "metrics": global_metrics,
    "per_category_summary": {k: v for k, v in sorted_cats} if per_category else {},
    "samples": [
        {
            "image_path": p,
            "category": c,
            "prediction": pred,
            "references": refs,
        }
        for p, c, pred, refs in zip(image_paths, categories, predictions, references)
    ],
}

output_file = os.path.join(OUTPUT_DIR, "eval_results.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(eval_result, f, ensure_ascii=False, indent=2)
print(f"  结果已保存: {output_file}")

# ── 打印样例预测 ──
print("\n" + "=" * 60)
print("  样例预测 (前 10 个)")
print("=" * 60)
for i in range(min(10, len(predictions))):
    print(f"\n  [{categories[i]}]")
    print(f"    Ref: {references[i][0][:120]}")
    print(f"    Pred: {predictions[i][:120]}")

print("\n" + "=" * 60)
print("  评估完成!")
print("=" * 60)
