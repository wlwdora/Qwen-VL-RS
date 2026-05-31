# 遥感图像描述 —— 训练模块
# 基于 HuggingFace Trainer 手写训练循环（不用 MS-SWIFT 黑盒）

from .trainer import RemoteSensingCaptionTrainer
from .loss import CaptioningLoss
from .metrics import CaptioningMetrics

__all__ = [
    "RemoteSensingCaptionTrainer",
    "CaptioningLoss",
    "CaptioningMetrics",
]
