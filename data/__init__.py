# 遥感图像描述 —— 数据处理模块
# 负责 RSICD / UCM-Captions / Sydney-Captions 数据集加载与预处理

from .dataset import RemoteSensingCaptionDataset
from .transforms import RemoteSensingTransforms
from .collator import MultiModalDataCollator
from .prompts import REMOTE_SENSING_PROMPTS

__all__ = [
    "RemoteSensingCaptionDataset",
    "RemoteSensingTransforms",
    "MultiModalDataCollator",
    "REMOTE_SENSING_PROMPTS",
]
