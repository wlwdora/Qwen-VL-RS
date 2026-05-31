"""RSICD LoRA 训练脚本 — 独立入口，避免模块导入 segfault。"""

import sys, os, gc, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
torch.cuda.empty_cache()
gc.collect()

# ── 延迟导入（避免一次性加载全部模块导致 segfault）───
from data.dataset import RemoteSensingCaptionDataset
from data.collator import MultiModalDataCollator
from data.transforms import RemoteSensingTransforms
from models.qwen_vl_rs import QwenVLForRemoteSensing
from training.loss import CaptioningLoss

# ── 配置 ────────────────────────────────
MODEL_PATH = "D:/Qwen/Qwen3-VL-2B-Instruct"
DATA_PATH = "D:/work/Qwen-VL-RS/data/processed/rsicd.jsonl"
OUTPUT_DIR = "D:/work/Qwen-VL-RS/output/qwen_vl_rs_lora"

CONFIG = {
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "learning_rate": 2e-5,
    "n_epochs": 3,
    "batch_size": 2,
    "grad_accum_steps": 4,
    "max_length": 512,
    "image_size": 512,
    "warmup_steps": 200,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "log_every": 10,
    "save_every": 500,
    "eval_every": 1000,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "logs"))

print("=" * 60)
print("  Qwen-VL-RS LoRA 训练 — RSICD")
print("=" * 60)
print(f"  Rank: {CONFIG['lora_rank']}, LR: {CONFIG['learning_rate']}")
print(f"  Epochs: {CONFIG['n_epochs']}, Batch: {CONFIG['batch_size']}")
print(f"  Grad Accum: {CONFIG['grad_accum_steps']}")
print("=" * 60)

# ════════════════════════════════════════════════════════════
# 1. 加载模型
# ════════════════════════════════════════════════════════════
print("\n[1/4] 加载模型...")
wrapper = QwenVLForRemoteSensing(
    model_path=MODEL_PATH,
    lora_config={
        "rank": CONFIG["lora_rank"],
        "alpha": CONFIG["lora_alpha"],
        "dropout": CONFIG["lora_dropout"],
    },
    torch_dtype=torch.float16,
)
wrapper.load()
wrapper.print_trainable_parameters()
model = wrapper.peft_model
model.train()

# ════════════════════════════════════════════════════════════
# 2. 加载数据
# ════════════════════════════════════════════════════════════
print("\n[2/4] 加载数据...")
train_transform = RemoteSensingTransforms(
    mode="train", image_size=CONFIG["image_size"]
).build()
eval_transform = RemoteSensingTransforms(
    mode="eval", image_size=CONFIG["image_size"]
).build()

train_ds = RemoteSensingCaptionDataset(
    data_paths=DATA_PATH, split="train",
    transform=train_transform, max_length=CONFIG["max_length"],
)
eval_ds = RemoteSensingCaptionDataset(
    data_paths=DATA_PATH, split="val",
    transform=eval_transform, max_length=CONFIG["max_length"],
)
print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")
print(f"  Categories: {len(train_ds.category_counts)}")
print(f"  Top categories: {list(train_ds.category_counts.items())[:5]}")

# ════════════════════════════════════════════════════════════
# 3. Collator & Loss & Optimizer
# ════════════════════════════════════════════════════════════
print("\n[3/4] 构建训练组件...")
collator = MultiModalDataCollator(
    tokenizer=wrapper.tokenizer,
    processor=wrapper.processor,
    max_length=CONFIG["max_length"],
    prompt="请详细描述这张遥感图像的内容。",
)
loss_fn = CaptioningLoss(loss_type="cross_entropy", ignore_index=-100)
device = next(model.parameters()).device

# Optimizer + Scheduler
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)
total_steps = (len(train_ds) // CONFIG["batch_size"]) * CONFIG["n_epochs"] // CONFIG["grad_accum_steps"]
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG["learning_rate"],
    total_steps=total_steps,
    pct_start=0.1,
)

# ════════════════════════════════════════════════════════════
# 4. 训练循环
# ════════════════════════════════════════════════════════════
print("\n[4/4] 开始训练")
print("-" * 60)

global_step = 0
best_eval_loss = float("inf")

for epoch in range(CONFIG["n_epochs"]):
    epoch_loss = 0.0
    epoch_start = time.time()
    n_batches = 0
    optimizer.zero_grad()

    # Shuffle
    indices = torch.randperm(len(train_ds))

    for batch_start in range(0, len(train_ds) - CONFIG["batch_size"] + 1, CONFIG["batch_size"]):
        end = min(batch_start + CONFIG["batch_size"], len(train_ds))
        batch_indices = indices[batch_start:end]

        # Build batch
        samples = [train_ds[int(i)] for i in batch_indices]
        try:
            batch = collator(samples)
        except Exception as e:
            print(f"  [WARN] Collator error at batch {batch_start}: {e}")
            continue

        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)

        # Forward
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fn(shift_logits, shift_labels) / CONFIG["grad_accum_steps"]

        # Backward
        loss.backward()

        step_loss = loss.item() * CONFIG["grad_accum_steps"]
        epoch_loss += step_loss
        n_batches += 1

        if (n_batches) % CONFIG["grad_accum_steps"] == 0:
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging
            if global_step % CONFIG["log_every"] == 0:
                lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / max(n_batches, 1)
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                print(f"  Epoch {epoch+1}/{CONFIG['n_epochs']} | "
                      f"Step {global_step:5d} | "
                      f"Loss {avg_loss:.4f} | "
                      f"LR {lr:.2e} | "
                      f"GPU {gpu_mem:.1f}G")
                writer.add_scalar("train/loss", step_loss, global_step)
                writer.add_scalar("train/lr", lr, global_step)

            # Save checkpoint
            if global_step % CONFIG["save_every"] == 0:
                ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                wrapper.save_lora(ckpt_dir)
                print(f"  [Checkpoint] Step {global_step} saved to {ckpt_dir}")

            # Eval
            if global_step % CONFIG["eval_every"] == 0:
                model.eval()
                eval_losses = []
                n_eval = min(200, len(eval_ds))  # 抽样评估，节省时间
                with torch.no_grad():
                    for i in range(0, n_eval - CONFIG["batch_size"] + 1, CONFIG["batch_size"]):
                        e_samples = [eval_ds[int(idx)] for idx in range(i, min(i + CONFIG["batch_size"], n_eval))]
                        try:
                            e_batch = collator(e_samples)
                        except:
                            continue
                        e_inputs = {k: v.to(device) for k, v in e_batch.items() if k != "labels"}
                        e_labels = e_batch["labels"].to(device)
                        e_outputs = model(**e_inputs)
                        e_logits = e_outputs.logits
                        e_shift_logits = e_logits[..., :-1, :].contiguous()
                        e_shift_labels = e_labels[..., 1:].contiguous()
                        e_loss = loss_fn(e_shift_logits, e_shift_labels)
                        eval_losses.append(e_loss.item())
                avg_eval_loss = sum(eval_losses) / max(len(eval_losses), 1)
                print(f"  [Eval] Step {global_step} | Eval Loss {avg_eval_loss:.4f}")
                writer.add_scalar("eval/loss", avg_eval_loss, global_step)

                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    wrapper.save_lora(os.path.join(OUTPUT_DIR, "best_model"))
                    print(f"  [Best] New best eval loss: {best_eval_loss:.4f}")

                model.train()

    # ── Epoch summary ─────────────────
    avg_epoch_loss = epoch_loss / max(n_batches, 1)
    elapsed = time.time() - epoch_start
    print(f"\n  === Epoch {epoch+1}/{CONFIG['n_epochs']} complete "
          f"| Loss {avg_epoch_loss:.4f} "
          f"| Time {elapsed/60:.1f}min ===\n")

    # Save per epoch
    wrapper.save_lora(os.path.join(OUTPUT_DIR, f"checkpoint-epoch{epoch+1}"))

# ════════════════════════════════════════════════════════════
# 完成
# ════════════════════════════════════════════════════════════
wrapper.save_lora(os.path.join(OUTPUT_DIR, "lora_adapter"))
writer.close()

print("=" * 60)
print(f"  训练完成!")
print(f"  最优 eval loss: {best_eval_loss:.4f}")
print(f"  模型保存在: {OUTPUT_DIR}")
print("=" * 60)

# Save config
with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2, ensure_ascii=False)
