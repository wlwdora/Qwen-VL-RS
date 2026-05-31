"""
遥感专用图像增强策略。

设计要点：
  1. 遥感图像没有固定朝向 → 离散旋转（0/90/180/270 度）
  2. 多尺度裁剪模拟不同地面采样距离（GSD）
  3. 光谱扰动模拟大气/光照条件变化
  4. 避免过激的颜色抖动用，以免破坏光谱信息的物理意义
"""

from typing import Dict, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


class RemoteSensingTransforms:
    """可组合的遥感图像变换管线。"""

    def __init__(
        self,
        mode: str = "train",  # "train" 或 "eval"
        image_size: int = 512,
        use_augmentation: bool = True,
        augmentation_config: Optional[Dict] = None,
    ):
        self.mode = mode
        self.image_size = image_size
        self.use_augmentation = use_augmentation and (mode == "train")

    def build(self) -> A.Compose:
        """构建变换管线。

        返回：
            Albumentations Compose 对象。
        """
        # TODO: 实现变换管线
        # 1. 保持宽高比的缩放
        # 2. 训练模式：离散旋转 + 翻转
        # 3. 训练模式：光谱扰动
        # 4. 归一化（CLIP 均值/标准差）
        # 5. ToTensorV2
        raise NotImplementedError("【待实现】变换管线")
