"""遥感图像描述的自定义损失函数。

标准交叉熵是默认选择，但我们提供以下替代方案：
  - Label Smoothing CE：标签平滑交叉熵，避免模型过度自信，提升泛化能力
  - Focal Loss：降低简单 token（如背景描述常用词）的权重，聚焦难例和稀有词
  - 对比描述损失（实验性）：拉近正样本描述的嵌入向量距离
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptioningLoss(nn.Module):
    """图像描述训练的统一损失函数模块。"""

    def __init__(
        self,
        loss_type: str = "cross_entropy",
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        ignore_index: int = -100,
    ):
        """
        参数：
            loss_type: 损失类型 —— "cross_entropy" | "label_smoothing" | "focal"
            label_smoothing: 标签平滑系数（0.0 = 不平滑）。
            focal_gamma: Focal Loss 的 gamma 参数（聚焦难例的程度）。
            focal_alpha: Focal Loss 的 alpha 参数（类别平衡系数）。
            ignore_index: 计算损失时忽略的 token id。
        """
        super().__init__()
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        参数：
            logits: (B, seq_len, vocab_size) 模型输出的 logits。
            labels: (B, seq_len) 真实 token id 序列。

        返回：
            标量损失值。
        """
        # TODO: 实现损失计算
        # - cross_entropy：标准 F.cross_entropy
        # - label_smoothing：带 label_smoothing 参数的交叉熵
        # - focal：计算 pt，施加 (1-pt)^gamma 权重
        raise NotImplementedError("【待实现】损失函数")
