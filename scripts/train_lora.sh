#!/bin/bash
# ==================== LoRA 微调脚本 ====================
# 用法：bash scripts/train_lora.sh [--config configs/sft_config.yaml]
set -e

echo "========================================"
echo "Qwen-VL-RS：LoRA 微调训练"
echo "========================================"

# 默认配置
CONFIG_PATH="${1:-configs/sft_config.yaml}"

# 启动训练
python -m training.trainer \
    --config "$CONFIG_PATH" \
    --mode train \
    2>&1 | tee experiments/logs/train_$(date +%Y%m%d_%H%M%S).log

echo "========================================"
echo "训练完成！"
echo "========================================"
