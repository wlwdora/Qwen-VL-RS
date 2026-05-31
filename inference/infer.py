"""单张图像和批量推理接口。"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class InferenceEngine:
    """加载训练好的模型并执行推理。"""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        """从 checkpoint 加载模型和处理器。"""
        # TODO: 实现推理模型加载
        raise NotImplementedError("【待实现】推理加载")

    def predict(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """为单张图像生成描述。

        参数：
            image: 图像路径或 PIL Image 对象。
            prompt: 自定义指令文本（默认使用遥感专家 prompt）。
            max_new_tokens: 最大生成长度。
            temperature: 采样温度。

        返回：
            生成的描述文本。
        """
        # TODO: 实现单图推理
        raise NotImplementedError("【待实现】单图推理")

    def predict_batch(
        self,
        images: List[Union[str, Image.Image]],
        prompts: Optional[List[str]] = None,
        batch_size: int = 4,
        **kwargs,
    ) -> List[str]:
        """为一个批次的图像生成描述。

        参数：
            images: 图像路径或 PIL Image 对象列表。
            prompts: 每张图对应的指令文本（可选）。
            batch_size: 推理批次大小。

        返回：
            描述文本列表。
        """
        # TODO: 实现批量推理
        raise NotImplementedError("【待实现】批量推理")
