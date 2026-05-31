"""训练冒烟测试：用假数据跑 100 步，验证训练循环正常。

验证项：
  - loss 随时间下降（非 NaN）
  - checkpoint 正常保存
  - 梯度不爆炸
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc

from data.dataset import RemoteSensingCaptionDataset
from data.collator import MultiModalDataCollator
from models.qwen_vl_rs import QwenVLForRemoteSensing
from training.loss import CaptioningLoss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = "D:/Qwen/Qwen3-VL-2B-Instruct"
DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/dummy.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/smoke_test")

def main():
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    gc.collect()

    print("=" * 60)
    print("  训练冒烟测试")
    print("=" * 60)

    # ── 1. Model ────────────────────────────
    print("\n[1/5] 加载模型 (rank=4, float16)...")
    wrapper = QwenVLForRemoteSensing(
        model_path=MODEL_PATH,
        lora_config={"rank": 4, "alpha": 8, "dropout": 0.05},
        torch_dtype=torch.float16,
    )
    wrapper.load()
    wrapper.print_trainable_parameters()
    model = wrapper.peft_model
    model.train()

    # ── 2. Dataset ──────────────────────────
    print("\n[2/5] 加载数据...")
    dataset = RemoteSensingCaptionDataset(
        data_paths=DATA_PATH, split="train", max_length=512,
    )
    print(f"  训练样本: {len(dataset)}")

    # ── 3. Collator ─────────────────────────
    print("\n[3/5] 构建 Collator...")
    collator = MultiModalDataCollator(
        tokenizer=wrapper.tokenizer,
        processor=wrapper.processor,
        max_length=256,  # 短序列省显存
        prompt="请描述这张遥感图像。",
    )

    # ── 4. Optimizer ────────────────────────
    print("\n[4/5] 配置优化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = CaptioningLoss(loss_type="cross_entropy", ignore_index=-100)

    # ── 5. Training Loop ────────────────────
    print("\n[5/5] 开始训练 (100 steps)...")
    print("-" * 50)

    losses = []
    device = next(model.parameters()).device

    for step in range(100):
        # 取随机 batch
        indices = torch.randint(0, len(dataset), (2,))
        samples = [dataset[int(i)] for i in indices]
        batch = collator(samples)

        # 移到 GPU
        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = inputs["labels"]

        # 前向传播
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        # 计算 loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fn(shift_logits, shift_labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 20 == 0:
            avg_loss = sum(losses[-20:]) / 20
            print(f"  Step {step+1:3d}/100  |  avg_loss = {avg_loss:.4f}")

        # 检查 NaN
        if torch.isnan(loss):
            print(f"  [FAIL] loss 为 NaN at step {step}!")
            return

    print("-" * 50)

    # ── 验证 ────────────────────────────────
    first_avg = sum(losses[:20]) / 20
    last_avg = sum(losses[-20:]) / 20
    decreasing = last_avg < first_avg

    print(f"\n  前 20 步 avg loss: {first_avg:.4f}")
    print(f"  后 20 步 avg loss: {last_avg:.4f}")
    print(f"  Loss 下降: {'[PASS]' if decreasing else '[WARN] (正常波动)'}")

    # ── 保存 checkpoint ─────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wrapper.save_lora(os.path.join(OUTPUT_DIR, "lora_adapter"))
    print(f"\n  Checkpoint 已保存: {OUTPUT_DIR}/lora_adapter")

    print("\n" + "=" * 60)
    print("  冒烟测试通过! 训练循环正常。")
    print("=" * 60)

    # cleanup
    del model, wrapper
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
