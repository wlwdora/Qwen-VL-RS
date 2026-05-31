"""遥感图像描述的自定义损失函数。

标准交叉熵是默认选择，但我们提供以下替代方案：
  - Label Smoothing CE：标签平滑交叉熵，避免模型过度自信，提升泛化能力
  - Focal Loss：降低简单 token（如背景描述常用词）的权重，聚焦难例和稀有词

使用方式：
    >>> loss_fn = CaptioningLoss(loss_type="focal", focal_gamma=2.0)
    >>> loss = loss_fn(logits, labels)  # logits: (B, seq_len, vocab), labels: (B, seq_len)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptioningLoss(nn.Module):
    """图像描述训练的统一损失函数模块。

    三种模式：
      - "cross_entropy": 标准交叉熵（默认）
      - "label_smoothing": 带标签平滑的交叉熵（平滑度由 label_smoothing 控制）
      - "focal": Focal Loss（聚焦难例，gamma 越大越关注难分类的 token）

    所有模式都支持 ignore_index（通常为 -100），用于忽略 padding 和 prompt token。
    """

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
            label_smoothing: 标签平滑系数（0.0 = 不平滑，推荐 0.1）。
                             用于 cross_entropy 和 label_smoothing 模式。
            focal_gamma: Focal Loss 的 gamma 参数（聚焦难例的程度，默认 2.0）。
            focal_alpha: Focal Loss 的 alpha 参数（类别平衡系数，默认 0.25）。
            ignore_index: 计算损失时忽略的 token id（默认为 -100，与 HuggingFace 对齐）。
        """
        super().__init__()
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.ignore_index = ignore_index

        # 内部 CE（用于 label_smoothing 模式）
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        参数：
            logits: (B, seq_len, vocab_size) 模型输出的 logits。
            labels: (B, seq_len) 真实 token id 序列（prompt 部分已用 ignore_index 掩码）。

        返回：
            标量损失值。
        """
        if self.loss_type == "cross_entropy":
            return self._cross_entropy(logits, labels)
        elif self.loss_type == "label_smoothing":
            return self._label_smoothing_ce(logits, labels)
        elif self.loss_type == "focal":
            return self._focal_loss(logits, labels)
        else:
            raise ValueError(
                f"不支持的 loss_type：'{self.loss_type}'。"
                f"可选：cross_entropy | label_smoothing | focal"
            )

    # ════════════════════════════════════════════════════════════
    # 各损失实现
    # ════════════════════════════════════════════════════════════

    def _cross_entropy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """标准交叉熵损失。

        将 (B, seq_len, vocab) 展平为 (B*seq_len, vocab)，与 (B*seq_len) 的 labels 对齐。
        """
        # logits: (B, seq_len, V) → (B*seq_len, V)
        # labels: (B, seq_len) → (B*seq_len)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,  # PyTorch 1.10+ 原生支持
        )

    def _label_smoothing_ce(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """标签平滑交叉熵。

        与 _cross_entropy 的区别：
          - 当 label_smoothing > 0 时，行为完全相同（都用 PyTorch 原生实现）
          - 此方法保留用于显式场景（如需要自定义平滑逻辑时修改）
        """
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

    def _focal_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Focal Loss 实现。

        核心公式：
            FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

        其中：
            p_t = softmax(logits) 在真实类别上的概率
            (1 - p_t)^γ：调制因子，预测越准（p_t 高）权重越低

        优势：自动降低简单 token（高频描述词如"有"、"是"）的梯度贡献，
              让模型更专注于学习稀有地物词汇和领域术语。

        参考资料：Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
        """
        # ── 计算 log probabilities ──────────────────
        log_probs = F.log_softmax(logits, dim=-1)  # (B, seq_len, V)

        # ── 收集真实类别的 log prob ────────────────
        # gather: 在 vocab 维度上取出 labels 对应的 log prob
        # labels: (B, seq_len) → (B, seq_len, 1) → gather → (B, seq_len, 1) → squeeze
        per_token_log_probs = log_probs.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, seq_len)

        # ── 计算概率和调制因子 ─────────────────────
        per_token_probs = per_token_log_probs.exp()  # p_t
        focal_weight = (1 - per_token_probs) ** self.focal_gamma  # (1-p_t)^γ

        # ── Alpha 加权（可选） ──────────────────────
        if self.focal_alpha is not None and self.focal_alpha > 0:
            # 对真实类别 token 施加 alpha 权重
            alpha_weight = torch.where(
                labels != self.ignore_index,
                self.focal_alpha,
                1.0,
            )
            focal_weight = focal_weight * alpha_weight.to(focal_weight.device)

        # ── 计算损失 ──────────────────────────────
        loss = -focal_weight * per_token_log_probs  # (B, seq_len)

        # ── 掩码 ignore_index ─────────────────────
        if self.ignore_index is not None:
            mask = (labels != self.ignore_index).float()
            loss = loss * mask
            # 用 sum / count 方式取平均（而非简单 mean，因为 ignore 位置不计入）
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss


# ════════════════════════════════════════════════════════════════
# 便捷构建函数
# ════════════════════════════════════════════════════════════════

def build_loss(
    loss_type: str = "cross_entropy",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    ignore_index: int = -100,
) -> CaptioningLoss:
    """工厂函数：一行创建损失函数。

    用法：
        >>> loss_fn = build_loss("cross_entropy", label_smoothing=0.1)
        >>> loss_fn = build_loss("focal", focal_gamma=2.0)
    """
    return CaptioningLoss(
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        ignore_index=ignore_index,
    )
