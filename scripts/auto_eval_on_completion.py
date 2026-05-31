"""训练完成后自动评测：检查训练是否结束，若结束则运行 eval 并保存结果。
用法：python scripts/auto_eval_on_completion.py
可配合 cron 定期调用。"""

import os, sys, json, time, subprocess
from pathlib import Path
from datetime import datetime

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJ_ROOT)

OUTPUT_DIR = Path("output/qwen_vl_rs_lora_r16_fixed")
TRAIN_LOG = Path("output/rsicd_r16_v2.log")
EVAL_LOG = Path("output/rsicd_r16_eval_fixed.log")
EVAL_DONE_MARKER = Path("output/.eval_completed")

BEST_MODEL = OUTPUT_DIR / "best_model"
PYTHON = "D:/VScode/anada/envs/torch/python.exe"
BASE_MODEL = "D:/Qwen/Qwen3-VL-2B-Instruct"

def is_training_done():
    """判断训练是否已完成（正常结束或早停触发）"""
    if not TRAIN_LOG.exists():
        return False
    text = TRAIN_LOG.read_text(encoding="utf-8", errors="ignore")
    # 检测中文原文或终端乱码
    done_markers = ["训练完成", "Early Stop", "最优 eval loss",
                    "Epoch 3/3 complete", "Epoch.*complete.*Loss"]
    import re
    for marker in done_markers:
        if re.search(marker, text):
            return True
    # 额外检查：日志中包含 "Loss data saved" 说明训练已结束
    if "Loss data saved" in text:
        return True
    return False

def best_model_ready():
    """检查 best_model 是否存在"""
    return (BEST_MODEL / "adapter_model.safetensors").exists()

def is_eval_done():
    """检查评估是否已完成"""
    return EVAL_DONE_MARKER.exists()

def run_eval():
    """运行评估"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 训练已完成，启动自动评估...")

    cmd = [
        PYTHON, "-u", "-m", "evaluation.eval",
        "--base_model", BASE_MODEL,
        "--checkpoint", str(BEST_MODEL),
        "--datasets", "rsicd",
        "--batch_size", "1",
    ]

    with open(EVAL_LOG, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=5400)

    if proc.returncode == 0:
        EVAL_DONE_MARKER.write_text(datetime.now().isoformat())
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 评估完成 → {EVAL_LOG}")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 评估失败，exit code: {proc.returncode}")

if __name__ == "__main__":
    if is_eval_done():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 评估已完成，跳过")
        sys.exit(0)

    if not is_training_done():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 训练进行中，等待...")
        sys.exit(0)

    if not best_model_ready():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 训练结束但 best_model 不存在！")
        sys.exit(1)

    run_eval()
