#!/bin/bash
# ==================== 评估脚本 ====================
# 用法：bash scripts/eval.sh [--checkpoint output/checkpoint-xxx]
set -e

echo "========================================"
echo "Qwen-VL-RS：模型评估"
echo "========================================"

CHECKPOINT="${1:-output/qwen_vl_rs_lora/best}"

python -m evaluation.eval \
    --checkpoint "$CHECKPOINT" \
    --datasets rsicd,ucm,sydney \
    --metrics bleu,meteor,cider,rouge,spice \
    2>&1 | tee experiments/logs/eval_$(date +%Y%m%d_%H%M%S).log

echo "========================================"
echo "评估完成！"
echo "========================================"
