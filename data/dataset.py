"""
遥感图像描述数据集加载器。

支持的数据集：
  - RSICD（10,921 张图像，每张 5 句描述，30+ 类别）
  - UCM-Captions（2,100 张图像，每张 5 句描述，21 类别）
  - Sydney-Captions（613 张图像，每张 5 句描述，7 类别）

数据格式（JSONL）：
  {"image": "路径/到/图片.jpg", "captions": ["...", "..."], "category": "机场"}
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


class RemoteSensingCaptionDataset(Dataset):
    """多源遥感图像描述数据集加载器。

    支持跨数据集加载、训练/验证/测试划分、
    以及可选的类别均衡采样。
    """

    def __init__(
        self,
        data_paths: Union[str, List[str]],
        split: str = "train",
        transform: Optional[object] = None,
        max_length: int = 512,
    ):
        """
        参数：
            data_paths: JSONL 标注文件路径（单个或多个）。
            split: 数据集划分，取 "train" / "val" / "test"。
            transform: Albumentations 或 torchvision 变换管线。
            max_length: 描述文本的最大 token 长度。
        """
        self.data_paths = [data_paths] if isinstance(data_paths, str) else data_paths
        self.split = split
        self.transform = transform
        self.max_length = max_length

        self.samples: List[Dict] = []
        self._load_data()

        logger.info(f"已加载 {len(self.samples)} 条样本，split='{split}'")

    def _load_data(self):
        """加载并解析所有标注文件。"""
        # TODO: 实现 JSONL 解析、train/val/test 划分
        raise NotImplementedError("【待实现】数据集加载逻辑")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回单条样本，包含图像张量和 tokenized 描述。"""
        # TODO: 实现 __getitem__
        raise NotImplementedError("【待实现】__getitem__ 方法")

    def get_category_distribution(self) -> Dict[str, int]:
        """返回各类别样本数量统计，用于数据分析。"""
        # TODO: 实现类别分布统计
        raise NotImplementedError("【待实现】类别统计")
