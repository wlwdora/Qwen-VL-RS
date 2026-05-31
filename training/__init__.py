# 遥感图像描述 —— 训练模块
# 基于 HuggingFace Trainer 手写训练循环（不用 MS-SWIFT 黑盒）
#
# 注意：trainer.py 因其依赖链（→ sklearn → pandas → pyarrow）
# 在某些 Windows 环境下导入时会触发 segfault。使用延迟导入避免崩溃。

from .loss import CaptioningLoss
from .metrics import CaptioningMetrics


def get_trainer():
    """延迟导入训练器（避免 sklearn/pandas import 链在 Windows 上的 segfault）。"""
    from .trainer import RemoteSensingCaptionTrainer
    return RemoteSensingCaptionTrainer


__all__ = [
    "CaptioningLoss",
    "CaptioningMetrics",
    "get_trainer",
]
