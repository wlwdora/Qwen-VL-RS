"""DPO 偏好对齐训练脚本。

在 SFT 模型基础上，用偏好对 (chosen > rejected) 进一步优化，
使模型偏向详细区分性描述，远离模板化输出。

用法：
    python scripts/train_dpo.py
"""

import sys, os, gc, time, json
try:
    _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    _proj_root = os.getcwd()
sys.path.insert(0, _proj_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

torch.manual_seed(42)

from models.qwen_vl_rs import QwenVLForRemoteSensing
from training.dpo_loss import DPOLoss, compute_log_probs
from data.transforms import RemoteSensingTransforms

# ── 配置 ────────────────────────────────
MODEL_PATH = "D:/Qwen/Qwen3-VL-2B-Instruct"
# 用 v2 模型（数据优选 + visual LoRA 重训后）作为 SFT 参考
SFT_ADAPTER_PATH = "D:/work/Qwen-VL-RS/output/qwen_vl_rs_lora_v2_dedup/best_model"
PREF_DATA_PATH = "D:/work/Qwen-VL-RS/data/processed/rsicd_dpo_pairs.jsonl"
OUTPUT_DIR = "D:/work/Qwen-VL-RS/output/qwen_vl_rs_dpo"

CONFIG = {
    # DPO 参数
    "beta": 0.1,                 # DPO 温度
    "label_smoothing": 0.0,     # 标签平滑
    # 训练参数
    "learning_rate": 5e-6,      # DPO 用更小的 lr（在 SFT 基础上微调）
    "n_epochs": 2,
    "batch_size": 1,            # 每批 1 个偏好对（chosen + rejected 各一次前向）
    "grad_accum_steps": 8,      # 等效 batch=8
    "max_length": 512,
    "image_size": 512,
    "warmup_steps": 50,
    "weight_decay": 0.01,
    "max_grad_norm": 0.5,       # DPO 梯度裁剪更严格
    "log_every": 10,
    "save_every": 500,
    "eval_every": 500,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "logs"))


# ════════════════════════════════════════════════════════════
# Preference Dataset
# ════════════════════════════════════════════════════════════

class PreferenceDataset(Dataset):
    """DPO 偏好数据集。

    每个 item 返回：
        image (PIL), prompt (str), chosen (str), rejected (str), category (str)
    """

    def __init__(
        self,
        data_path: str,
        transform=None,
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        self.transform = transform
        self.split = split

        # 加载所有偏好对
        self.pairs = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.pairs.append(json.loads(line))

        # 划分
        n = len(self.pairs)
        n_train = int(n * train_ratio)
        indices = torch.randperm(n).tolist()

        if split == "train":
            self.pairs = [self.pairs[i] for i in indices[:n_train]]
        else:
            self.pairs = [self.pairs[i] for i in indices[n_train:]]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair["image"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "prompt": pair["prompt"],
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "category": pair["category"],
        }


# ════════════════════════════════════════════════════════════
# Collate function
# ════════════════════════════════════════════════════════════

def collate_dpo_batch(batch, processor, tokenizer, max_length):
    """将一个 batch 的偏好对编码为模型输入。

    对每个样本，分别编码 (image, prompt, chosen) 和 (image, prompt, rejected)。
    返回两个并行的 batch。
    """
    chosen_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    rejected_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    pixel_values_list = []
    image_grid_thw_list = []

    for item in batch:
        image = item["image"]
        prompt = item["prompt"]

        # ── 编码 chosen ──────────────────────
        chosen_text = prompt + "\n" + item["chosen"]
        chosen_encoded = _encode_single(processor, tokenizer, image, prompt, item["chosen"], max_length)

        # ── 编码 rejected ────────────────────
        rejected_text = prompt + "\n" + item["rejected"]
        rejected_encoded = _encode_single(processor, tokenizer, image, prompt, item["rejected"], max_length)

        chosen_inputs["input_ids"].append(chosen_encoded["input_ids"])
        chosen_inputs["attention_mask"].append(chosen_encoded["attention_mask"])
        chosen_inputs["labels"].append(chosen_encoded["labels"])
        rejected_inputs["input_ids"].append(rejected_encoded["input_ids"])
        rejected_inputs["attention_mask"].append(rejected_encoded["attention_mask"])
        rejected_inputs["labels"].append(rejected_encoded["labels"])
        pixel_values_list.append(chosen_encoded["pixel_values"])
        image_grid_thw_list.append(chosen_encoded["image_grid_thw"])

    # Pad
    def pad_and_stack(tensor_list, pad_value=0):
        max_len = max(t.shape[0] for t in tensor_list)
        padded = []
        for t in tensor_list:
            p = torch.full((max_len,), pad_value, dtype=t.dtype)
            p[:t.shape[0]] = t
            padded.append(p)
        return torch.stack(padded)

    batch_chosen = {
        "input_ids": pad_and_stack(chosen_inputs["input_ids"], pad_value=tokenizer.pad_token_id or 0),
        "attention_mask": pad_and_stack(chosen_inputs["attention_mask"]),
        "labels": pad_and_stack(chosen_inputs["labels"], pad_value=-100),
    }
    batch_rejected = {
        "input_ids": pad_and_stack(rejected_inputs["input_ids"], pad_value=tokenizer.pad_token_id or 0),
        "attention_mask": pad_and_stack(rejected_inputs["attention_mask"]),
        "labels": pad_and_stack(rejected_inputs["labels"], pad_value=-100),
    }

    # pixel_values 和 image_grid_thw 合并
    pv = torch.stack(pixel_values_list) if pixel_values_list else None
    igt = torch.stack(image_grid_thw_list) if image_grid_thw_list and image_grid_thw_list[0] is not None else None

    batch_chosen["pixel_values"] = pv
    batch_chosen["image_grid_thw"] = igt
    batch_rejected["pixel_values"] = pv
    batch_rejected["image_grid_thw"] = igt

    return batch_chosen, batch_rejected


def _encode_single(processor, tokenizer, image, prompt, caption, max_length):
    """编码单条 (image, prompt, caption) 为 input_ids + labels。"""
    # 用 chat template 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": caption}],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False)

    # Processor 编码
    processor_inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )

    # 构建 labels：mask 掉 prompt 部分
    full_text = processor_inputs["input_ids"][0]

    # 只编码 prompt 部分来定位 cutoff
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
        },
    ]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False)
    prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]

    labels = full_text.clone()
    labels[:prompt_len] = -100

    return {
        "input_ids": full_text,
        "attention_mask": torch.ones_like(full_text),
        "labels": labels,
        "pixel_values": processor_inputs["pixel_values"][0],
        "image_grid_thw": processor_inputs.get("image_grid_thw", [None])[0],
    }


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Qwen-VL-RS DPO 偏好对齐训练")
    print("=" * 60)
    print(f"  β={CONFIG['beta']}, LR={CONFIG['learning_rate']}")
    print(f"  Epochs={CONFIG['n_epochs']}, GradAccum={CONFIG['grad_accum_steps']}")
    print("=" * 60)

    # ── 1. 加载模型 ────────────────────────
    print("\n[1/5] 加载模型...")
    # 检查 SFT adapter 是否存在
    if os.path.exists(SFT_ADAPTER_PATH):
        print(f"  加载 SFT adapter: {SFT_ADAPTER_PATH}")
        wrapper = QwenVLForRemoteSensing.from_pretrained(
            model_path=MODEL_PATH,
            lora_adapter_path=SFT_ADAPTER_PATH,
            torch_dtype=torch.float16,
        )
    else:
        print("  [WARN] SFT adapter 不存在，使用基座模型")
        wrapper = QwenVLForRemoteSensing(
            model_path=MODEL_PATH,
            lora_config={
                "rank": 32,
                "alpha": 64,
                "dropout": 0.05,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "qkv", "proj", "linear_fc1", "linear_fc2",
                ],
            },
            torch_dtype=torch.float16,
        )
        wrapper.load()

    wrapper.print_trainable_parameters()
    policy_model = wrapper.peft_model
    device = next(policy_model.parameters()).device

    # ── 参考模型（冻结的 SFT 模型副本）──
    # 简便做法：clone policy 作为 reference，但更省显存的做法是共享基座权重
    # 这里用 deepcopy policy 的方式（简单但占显存 ~4GB × 2）
    import copy
    print("  克隆参考模型（冻结）...")
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print("  参考模型已冻结")

    # ── 2. 加载偏好数据 ────────────────────
    print("\n[2/5] 加载偏好数据...")
    transform = RemoteSensingTransforms(
        mode="train", image_size=CONFIG["image_size"]
    ).build()

    train_ds = PreferenceDataset(
        data_path=PREF_DATA_PATH,
        transform=transform,
        split="train",
        train_ratio=0.9,
    )
    val_ds = PreferenceDataset(
        data_path=PREF_DATA_PATH,
        transform=transform,
        split="val",
        train_ratio=0.9,
    )
    print(f"  Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_dpo_batch(
            b, wrapper.processor, wrapper.tokenizer, CONFIG["max_length"]
        ),
    )

    # ── 3. Loss & Optimizer ─────────────────
    print("\n[3/5] 构建训练组件...")
    dpo_loss_fn = DPOLoss(
        beta=CONFIG["beta"],
        label_smoothing=CONFIG["label_smoothing"],
    )

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    total_steps = len(train_loader) * CONFIG["n_epochs"] // CONFIG["grad_accum_steps"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["learning_rate"],
        total_steps=total_steps,
        pct_start=0.1,
    )

    # ── 4. 训练循环 ─────────────────────────
    print("\n[4/5] 开始 DPO 训练")
    print(f"  Total steps: {total_steps}")
    print("-" * 60)

    global_step = 0
    best_accuracy = 0.0
    optimizer.zero_grad()

    for epoch in range(CONFIG["n_epochs"]):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_start = time.time()
        n_batches = 0

        for batch_idx, (batch_chosen, batch_rejected) in enumerate(train_loader):
            # 移到 GPU
            batch_chosen = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch_chosen.items()}
            batch_rejected = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch_rejected.items()}

            # ── Policy forward (chosen + rejected) ──
            policy_chosen_logps = compute_log_probs(
                policy_model,
                input_ids=batch_chosen["input_ids"],
                attention_mask=batch_chosen["attention_mask"],
                labels=batch_chosen["labels"],
                pixel_values=batch_chosen.get("pixel_values"),
                image_grid_thw=batch_chosen.get("image_grid_thw"),
            )
            policy_rejected_logps = compute_log_probs(
                policy_model,
                input_ids=batch_rejected["input_ids"],
                attention_mask=batch_rejected["attention_mask"],
                labels=batch_rejected["labels"],
                pixel_values=batch_rejected.get("pixel_values"),
                image_grid_thw=batch_rejected.get("image_grid_thw"),
            )

            # ── Reference forward (no grad) ──
            with torch.no_grad():
                ref_chosen_logps = compute_log_probs(
                    ref_model,
                    input_ids=batch_chosen["input_ids"],
                    attention_mask=batch_chosen["attention_mask"],
                    labels=batch_chosen["labels"],
                    pixel_values=batch_chosen.get("pixel_values"),
                    image_grid_thw=batch_chosen.get("image_grid_thw"),
                )
                ref_rejected_logps = compute_log_probs(
                    ref_model,
                    input_ids=batch_rejected["input_ids"],
                    attention_mask=batch_rejected["attention_mask"],
                    labels=batch_rejected["labels"],
                    pixel_values=batch_rejected.get("pixel_values"),
                    image_grid_thw=batch_rejected.get("image_grid_thw"),
                )

            # ── DPO Loss ──
            loss_outputs = dpo_loss_fn(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
            )
            loss = loss_outputs["loss"] / CONFIG["grad_accum_steps"]
            loss.backward()

            epoch_loss += loss_outputs["loss"].item()
            epoch_acc += loss_outputs["accuracy"]
            n_batches += 1

            # ── Gradient accumulation ──
            if n_batches % CONFIG["grad_accum_steps"] == 0:
                nn.utils.clip_grad_norm_(
                    policy_model.parameters(), CONFIG["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % CONFIG["log_every"] == 0:
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / n_batches
                    avg_acc = epoch_acc / n_batches
                    margin = loss_outputs["reward_margin"]
                    elapsed = (time.time() - epoch_start) / 60
                    print(
                        f"  Epoch {epoch+1}/{CONFIG['n_epochs']} | "
                        f"Step {global_step:5d} | "
                        f"Loss {avg_loss:.4f} | "
                        f"Acc {avg_acc:.2f} | "
                        f"Margin {margin:+.3f} | "
                        f"LR {lr:.2e} | "
                        f"Elapsed {elapsed:.0f}min"
                    )
                    writer.add_scalar("dpo/loss", loss_outputs["loss"].item(), global_step)
                    writer.add_scalar("dpo/accuracy", loss_outputs["accuracy"], global_step)
                    writer.add_scalar("dpo/reward_margin", margin, global_step)

                if global_step % CONFIG["save_every"] == 0:
                    ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    wrapper.save_lora(ckpt_dir)
                    print(f"  [Checkpoint] Step {global_step} saved")

            # 显存清理
            del batch_chosen, batch_rejected
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # ── Epoch summary ──
        avg_epoch_loss = epoch_loss / n_batches
        avg_epoch_acc = epoch_acc / n_batches
        elapsed = time.time() - epoch_start
        print(
            f"\n  === Epoch {epoch+1}/{CONFIG['n_epochs']} complete "
            f"| Loss {avg_epoch_loss:.4f} "
            f"| Acc {avg_epoch_acc:.2f} "
            f"| Time {elapsed/60:.1f}min ===\n"
        )

        wrapper.save_lora(os.path.join(OUTPUT_DIR, f"checkpoint-epoch{epoch+1}"))

        # 保存最佳
        if avg_epoch_acc > best_accuracy:
            best_accuracy = avg_epoch_acc
            wrapper.save_lora(os.path.join(OUTPUT_DIR, "best_model"))

    # ── 5. 完成 ────────────────────────────
    print("\n[5/5] 保存最终模型...")
    wrapper.save_lora(os.path.join(OUTPUT_DIR, "lora_adapter"))
    writer.close()

    with open(os.path.join(OUTPUT_DIR, "dpo_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"  DPO 训练完成!")
    print(f"  最佳 accuracy: {best_accuracy:.3f}")
    print(f"  模型保存在: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
