"""端到端管线验证脚本。

验证：Dataset → Collator → 模型前向传播 → Loss 计算
不执行完整训练，只确认数据流正确、loss 不炸。
"""

import sys
import os

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from data.dataset import RemoteSensingCaptionDataset
from data.collator import MultiModalDataCollator
from training.loss import CaptioningLoss


def main():
    MODEL_PATH = "D:/Qwen/Qwen3-VL-2B-Instruct"
    DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/dummy.jsonl")

    print("=" * 60)
    print("  Qwen-VL-RS 端到端管线验证")
    print("=" * 60)

    # ── Step 1: Dataset ─────────────────────
    print("\n[1/5] 加载 Dataset...")
    dataset = RemoteSensingCaptionDataset(
        data_paths=DATA_PATH,
        split="train",
        max_length=512,
    )
    print(f"  样本数: {len(dataset)}")
    print(f"  类别数: {len(dataset.category_counts)}")

    # 测试 __getitem__
    sample = dataset[0]
    print(f"  sample keys: {list(sample.keys())}")
    print(f"  image type: {type(sample['pixel_values']).__name__}")
    print(f"  caption: {str(sample['captions'])[:80]}...")

    # ── Step 2: Model ───────────────────────
    print("\n[2/5] 加载模型...")
    from models.qwen_vl_rs import QwenVLForRemoteSensing

    wrapper = QwenVLForRemoteSensing(
        model_path=MODEL_PATH,
        lora_config={"rank": 4, "alpha": 8, "dropout": 0.05},  # 小 rank 省显存
        torch_dtype=torch.float16,  # CUDA 11.8 必须显式指定 float16
    )
    wrapper.load()

    # 打印可训练参数
    wrapper.print_trainable_parameters()

    # ── Step 3: Collator ────────────────────
    print("\n[3/5] 构建 Collator...")
    collator = MultiModalDataCollator(
        tokenizer=wrapper.tokenizer,
        processor=wrapper.processor,
        max_length=512,
        prompt="请详细描述这张遥感图像的内容。",
    )

    # 取一个小 batch
    batch_samples = [dataset[i] for i in range(4)]
    batch = collator(batch_samples)
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  pixel_values shape: {batch['pixel_values'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")

    # 检查 labels 掩码
    masked_ratio = (batch['labels'] == -100).float().mean().item()
    print(f"  labels 掩码比例 (prompt部分): {masked_ratio:.1%}")
    print(f"  image_grid_thw: {batch.get('image_grid_thw', 'N/A')}")

    # ── Step 4: Forward Pass ────────────────
    print("\n[4/5] 前向传播...")
    model = wrapper.peft_model or wrapper.model
    device = next(model.parameters()).device

    # 移到设备
    inputs = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    print(f"  logits shape: {logits.shape}")
    print(f"  logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

    if hasattr(outputs, "loss") and outputs.loss is not None:
        print(f"  model loss: {outputs.loss.item():.4f}")

    # ── Step 5: Custom Loss ─────────────────
    print("\n[5/5] 计算自定义 Loss...")
    labels = inputs["labels"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    for loss_type in ["cross_entropy", "focal"]:
        loss_fn = CaptioningLoss(
            loss_type=loss_type, label_smoothing=0.1, ignore_index=-100
        )
        loss = loss_fn(shift_logits, shift_labels)
        ok = "OK" if not torch.isnan(loss) else "FAIL (NaN)"
        print(f"  {loss_type}: {loss.item():.4f}  [{ok}]")

    # ── 总结 ────────────────────────────────
    print("\n" + "=" * 60)
    print("  验证通过！管线数据流正常。")
    print(f"  模型: {MODEL_PATH}")
    print(f"  数据: {DATA_PATH}")
    print(f"  设备: {device}")
    print("=" * 60)


if __name__ == "__main__":
    main()
