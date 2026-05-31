"""Zero-shot baseline 评测 — Qwen3-VL-2B-Instruct 无任何微调，直接推理。"""
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
DATA_PATH = "D:/work/Qwen-VL-RS/data/processed/rsicd.jsonl"
OUTPUT_DIR = "D:/work/Qwen-VL-RS/experiments/evaluations"
DEVICE = "cuda"

MAX_SAMPLES = 300  # zero-shot 抽 300 样本即可获得可靠基线
BATCH_SIZE = 1     # 逐样本生成以确保稳定性
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  Zero-Shot Baseline — Qwen3-VL-2B-Instruct (no LoRA)")
print("=" * 60)

# ════════════════════════════════════════════════════════════
# 1. 加载模型（不加 LoRA）
# ════════════════════════════════════════════════════════════
print("\n[1/4] 加载基座模型 (zero-shot, 无 LoRA)...")
wrapper = QwenVLForRemoteSensing(
    model_path=MODEL_PATH,
    lora_config=None,  # ← 不加 LoRA
    torch_dtype=torch.float16,
)
wrapper.load()
model = wrapper.model  # 直接用基座模型，不用 peft_model
model.eval()
wrapper.print_trainable_parameters()  # 应显示 0 trainable
print(f"  模型加载完成 (zero-shot)")

# ════════════════════════════════════════════════════════════
# 2. 加载数据
# ════════════════════════════════════════════════════════════
print("\n[2/4] 加载数据...")
test_ds = RemoteSensingCaptionDataset(
    data_paths=DATA_PATH, split="test", max_length=512,
)
n_total = len(test_ds)
n_eval = min(MAX_SAMPLES, n_total)
print(f"  Test set: {n_total} samples, evaluating {n_eval} samples")

# ════════════════════════════════════════════════════════════
# 3. Collator
# ════════════════════════════════════════════════════════════
print("\n[3/4] 构建 Collator...")
collator = MultiModalDataCollator(
    tokenizer=wrapper.tokenizer,
    processor=wrapper.processor,
    max_length=512,
    prompt="Describe this remote sensing image in detail.",
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

for i in range(n_eval):
    sample = test_ds[int(i)]
    try:
        batch = collator([sample])
    except Exception as e:
        print(f"  [WARN] Collator error at sample {i}: {e}")
        predictions.append("")
        references.append(sample["captions"])
        categories.append(sample["category"])
        image_paths.append(sample["image_path"])
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
            captions = [""]

    predictions.append(captions[0].strip() if len(captions) > 0 else "")
    references.append(sample["captions"])
    categories.append(sample["category"])
    image_paths.append(sample["image_path"])

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start_time
        speed = (i + 1) / elapsed
        eta = (n_eval - i - 1) / speed
        print(f"  [{i+1}/{n_eval}] {speed:.1f} samples/s, ETA {eta:.0f}s")

elapsed_total = time.time() - start_time
print(f"\n  推理完成! {len(predictions)} predictions in {elapsed_total:.1f}s ({len(predictions)/elapsed_total:.1f} samples/s)")

# ════════════════════════════════════════════════════════════
# 5. 计算指标
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  计算评估指标...")
print("=" * 60)

from training.metrics import CaptioningMetrics
metrics_calc = CaptioningMetrics(metrics=["bleu", "rouge", "cider"])
results = metrics_calc.compute_all_detailed(predictions, references, categories)
global_metrics = results["global"]
per_category = results.get("per_category", {})

print("\n" + metrics_calc.format_results(global_metrics))

if per_category:
    sorted_cats = sorted(per_category.items(), key=lambda x: x[1].get("bleu_4", 0), reverse=True)
    print(f"\n  Best 5 categories (by BLEU-4):")
    for cat, scores in sorted_cats[:5]:
        print(f"    {cat:25s}: BLEU-4={scores.get('bleu_4',0):.3f}, ROUGE-L={scores.get('rouge_l',0):.3f}, CIDEr={scores.get('cider_d',0):.2f}")
    print(f"\n  Worst 5 categories (by BLEU-4):")
    for cat, scores in sorted_cats[-5:]:
        print(f"    {cat:25s}: BLEU-4={scores.get('bleu_4',0):.3f}, ROUGE-L={scores.get('rouge_l',0):.3f}, CIDEr={scores.get('cider_d',0):.2f}")

# ════════════════════════════════════════════════════════════
# 6. 保存结果
# ════════════════════════════════════════════════════════════
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(OUTPUT_DIR, f"rsicd_zero_shot_{timestamp}.json")
eval_result = {
    "model": "Qwen3-VL-2B-Instruct (zero-shot, no LoRA)",
    "dataset": "RSICD",
    "n_samples": len(predictions),
    "metrics": global_metrics,
    "per_category_summary": {k: v for k, v in sorted_cats} if per_category else {},
    "samples": [
        {"image_path": p, "category": c, "prediction": pred, "references": refs}
        for p, c, pred, refs in zip(image_paths, categories, predictions, references)
    ],
}
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(eval_result, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {output_file}")

# ── 打印样例预测 ──
print("\n" + "=" * 60)
print("  Zero-Shot 样例预测 (前 10 个)")
print("=" * 60)
for i in range(min(10, len(predictions))):
    print(f"\n  [{categories[i]}]")
    print(f"    Ref: {references[i][0][:120]}")
    print(f"    Pred: {predictions[i][:120]}")

print("\n" + "=" * 60)
print("  Zero-shot baseline 完成!")
print("=" * 60)
