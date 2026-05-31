#!/bin/bash
# ==================== 全量 SFT 微调脚本 ====================
# 用法：bash scripts/train_sft.sh [--config configs/sft_config.yaml]
set -e

echo "========================================"
echo "Qwen-VL-RS：全量 SFT 微调（Baseline）"
echo "========================================"

CONFIG_PATH="${1:-configs/sft_config.yaml}"

python -m training.trainer \
    --config "$CONFIG_PATH" \
    --mode sft \
    2>&1 | tee experiments/logs/sft_$(date +%Y%m%d_%H%M%S).log

echo "========================================"
echo "SFT 训练完成！"
echo "========================================"
