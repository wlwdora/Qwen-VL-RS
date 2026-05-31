"""
遥感专用图像增强策略。

设计要点：
  1. 遥感图像没有固定朝向 → 离散旋转（0/90/180/270 度）
  2. 多尺度裁剪模拟不同地面采样距离（GSD）
  3. 光谱扰动模拟大气/光照条件变化
  4. 避免过激的颜色抖动，以免破坏光谱信息的物理意义
"""

from typing import Dict, Optional, Tuple, List
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


# CLIP 模型的标准归一化参数（Qwen-VL 系列也沿用此参数）
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class RemoteSensingTransforms:
    """可组合的遥感图像变换管线。

    使用方式：
        transforms = RemoteSensingTransforms(mode="train", image_size=512)
        pipeline = transforms.build()
        augmented = pipeline(image=np.array(pil_image))  # 输入 HWC uint8 numpy 数组
        tensor = augmented["image"]                       # 输出 CHW float32 torch.Tensor
    """

    def __init__(
        self,
        mode: str = "train",
        image_size: int = 512,
        use_augmentation: bool = True,
        augmentation_config: Optional[Dict] = None,
    ):
        """
        参数：
            mode: "train"（应用数据增强）或 "eval"（仅缩放+归一化）。
            image_size: 输出图像的边长（正方形）。
            use_augmentation: 训练模式下是否启用增强（可用于消融实验 A5）。
            augmentation_config: 可选的增强参数覆盖字典，支持以下 key：
                - rotation_angles: List[int]  离散旋转角度列表
                - hflip_p: float              水平翻转概率
                - vflip_p: float              垂直翻转概率
                - crop_scale: Tuple[float,float]  随机裁剪尺度范围
                - brightness: float           亮度扰动幅度
                - contrast: float             对比度扰动幅度
                - saturation: float           饱和度扰动幅度
                - hue: float                  色调扰动幅度
                - color_jitter_p: float       光谱扰动触发概率
        """
        self.mode = mode
        self.image_size = image_size
        self.use_augmentation = use_augmentation and (mode == "train")

        # 设置增强参数默认值，允许通过 config 覆盖
        cfg = augmentation_config or {}
        self.rotation_angles: List[int] = cfg.get("rotation", [0, 90, 180, 270])
        self.hflip_p: float = cfg.get("horizontal_flip", 0.5)
        self.vflip_p: float = cfg.get("vertical_flip", 0.5)
        self.crop_scale: Tuple[float, float] = tuple(cfg.get("random_crop_scale", [0.5, 1.5]))
        self.brightness: float = cfg.get("brightness", 0.1)
        self.contrast: float = cfg.get("contrast", 0.1)
        self.saturation: float = cfg.get("saturation", 0.1)
        self.hue: float = cfg.get("hue", 0.05)
        self.color_jitter_p: float = cfg.get("color_jitter_p", 0.5)

    def _build_resize(self) -> A.Compose:
        """构建缩放 + 填充管线（保持宽高比，pad 到正方形）。

        为什么不直接用 Resize(image_size, image_size)：
          - 直接拉伸会扭曲地物形状（如把圆形农田压成椭圆）
          - 保持宽高比后 pad 黑色区域，对模型来说黑边是"无信息区域"，
            不会误导空间理解
        """
        return A.Compose([
            A.LongestMaxSize(max_size=self.image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,  # albumentations 2.0+: value → fill
            ),
        ])

    def _build_augmentations(self) -> list:
        """构建遥感专用数据增强管线。"""
        transforms = []

        # ── 空间增强 ──────────────────────────────
        # 离散旋转：遥感图像无固定朝向，用离散而非连续是防止
        # 连续旋转引入的插值伪影被模型误认为地物纹理
        transforms.append(
            A.RandomRotate90()  # 等价于离散旋转 0/90/180/270 度
        )

        # 水平翻转
        if self.hflip_p > 0:
            transforms.append(A.HorizontalFlip(p=self.hflip_p))

        # 垂直翻转：航拍/卫星图像上下对称性普遍存在
        if self.vflip_p > 0:
            transforms.append(A.VerticalFlip(p=self.vflip_p))

        # 多尺度裁剪：模拟不同空间分辨率（GSD）下的成像效果
        # scale 范围 [0.5, 1.0] 意味着从原图的 50%~100% 区域裁剪后 resize
        # albumentations 2.0+: height/width → size, scale 必须在 [0,1]
        transforms.append(
            A.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=tuple(min(1.0, s) for s in self.crop_scale),  # 确保 ≤1
                ratio=(0.9, 1.1),  # 允许轻微宽高比变化
                interpolation=cv2.INTER_LINEAR,
                p=1.0,
            )
        )

        # ── 光谱增强 ──────────────────────────────
        # 使用小幅度扰动模拟：
        #   - 亮度变化 → 不同太阳高度角/云量
        #   - 对比度变化 → 大气散射/雾霾
        #   - 饱和度变化 → 传感器差异
        #   - 色调变化 → 波段偏移（极小幅度，保持物理合理性）
        if any([self.brightness, self.contrast, self.saturation, self.hue]):
            transforms.append(
                A.ColorJitter(
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue,
                    p=self.color_jitter_p,
                )
            )

        return transforms

    def build(self, return_as_tensor: bool = True) -> A.Compose:
        """构建变换管线。

        参数：
            return_as_tensor: 是否在管线末尾附加 Normalize + ToTensorV2。
                             设为 False 可输出 numpy 数组，交由 processor 处理。

        返回：
            Albumentations Compose 对象。
        """
        layers = []

        # ── 第 1 步：统一尺寸（训练和评估都需要） ──
        layers.append(self._build_resize())

        # ── 第 2 步：数据增强（仅训练模式） ──
        if self.use_augmentation:
            layers.extend(self._build_augmentations())

        # ── 第 3 步：归一化 + 转张量 ──
        if return_as_tensor:
            layers.append(A.Normalize(mean=CLIP_MEAN, std=CLIP_STD))
            layers.append(ToTensorV2())

        return A.Compose(layers)

    def build_separate(self) -> Tuple[A.Compose, Optional[A.Compose]]:
        """分别构建"共享预处理"和"纯增强"管线。

        这个 split 设计服务于推理场景：
          - 共享预处理：eval 时和 train 时都要做 resize + pad
          - 增强管线：仅 train 时需要

        返回：
            (shared_pipeline, augmentation_pipeline)
        """
        shared = A.Compose([
            A.LongestMaxSize(max_size=self.image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
            ),
            A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ToTensorV2(),
        ])

        if self.use_augmentation:
            aug = A.Compose(
                self._build_augmentations() + [
                    A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
                    ToTensorV2(),
                ]
            )
        else:
            aug = None

        return shared, aug


def get_transforms(
    mode: str = "train",
    image_size: int = 512,
    augmentation_config: Optional[Dict] = None,
) -> A.Compose:
    """便捷工厂函数——一行获取完整变换管线。

    用法：
        >>> train_transforms = get_transforms("train", 512)
        >>> eval_transforms = get_transforms("eval", 512)
    """
    return RemoteSensingTransforms(
        mode=mode,
        image_size=image_size,
        augmentation_config=augmentation_config,
    ).build()
