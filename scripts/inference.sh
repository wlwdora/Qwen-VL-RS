#!/bin/bash
# ==================== 推理脚本 ====================
# 用法：bash scripts/inference.sh [--image 路径/到/图片.jpg]
set -e

echo "========================================"
echo "Qwen-VL-RS：推理"
echo "========================================"

IMAGE_PATH="${1}"

if [ -z "$IMAGE_PATH" ]; then
    echo "未指定图像，启动 Gradio 交互 Demo..."
    python -m inference.gradio_app
else
    echo "对图像进行推理：$IMAGE_PATH"
    python -m inference.infer --image "$IMAGE_PATH"
fi
