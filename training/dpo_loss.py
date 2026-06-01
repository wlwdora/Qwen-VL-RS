"""DPO (Direct Preference Optimization) Loss。

DPO 直接优化偏好数据，无需单独训练 reward model。

公式：
  L_DPO = -E[log σ(β * (log_ratio_chosen - log_ratio_rejected))]
  log_ratio = log π_θ(y|x) - log π_ref(y|x)

其中：
  π_θ  = 当前 policy（正在训练的模型）
  π_ref = 参考 policy（冻结的 SFT 模型）
  β    = 温度参数（控制偏离参考模型的程度）

参考资料：Rafailov et al., "Direct Preference Optimization", NeurIPS 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class DPOLoss(nn.Module):
    """DPO 损失函数。

    用法：
        >>> loss_fn = DPOLoss(beta=0.1)
        >>> loss, metrics = loss_fn(
        ...     policy_chosen_logps, policy_rejected_logps,
        ...     ref_chosen_logps, ref_rejected_logps,
        ... )
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        reference_free: bool = False,
    ):
        """
        参数：
            beta: DPO 温度。越小越保守（更贴近参考模型），越大越激进。
                  推荐范围 0.05-0.5，caption 任务建议 0.1。
            label_smoothing: 标签平滑（保守项，reduce over-optimization）。
            reference_free: True 时不需参考模型（等同于 IPO loss）。
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.reference_free = reference_free

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,      # (B,) 当前模型在 chosen 上的 log-prob
        policy_rejected_logps: torch.Tensor,    # (B,) 当前模型在 rejected 上的 log-prob
        ref_chosen_logps: Optional[torch.Tensor] = None,    # (B,) 参考模型在 chosen 上的 log-prob
        ref_rejected_logps: Optional[torch.Tensor] = None,  # (B,) 参考模型在 rejected 上的 log-prob
    ) -> Dict[str, float]:
        """计算 DPO loss。

        返回：
            {"loss": float, "accuracy": float, "chosen_rewards": float, "rejected_rewards": float}
        """
        if not self.reference_free:
            assert ref_chosen_logps is not None, "reference_free=False 时需要 ref_chosen_logps"
            assert ref_rejected_logps is not None, "reference_free=False 时需要 ref_rejected_logps"

        # ── 计算 log-ratio ──────────────────────────
        if self.reference_free:
            # IPO-style: 没有参考模型，直接对比
            log_ratio_chosen = policy_chosen_logps
            log_ratio_rejected = policy_rejected_logps
        else:
            log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
            log_ratio_rejected = policy_rejected_logps - ref_rejected_logps

        # ── DPO loss ─────────────────────────────────
        # loss = -log(σ(β * (log_ratio_chosen - log_ratio_rejected)))
        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)

        if self.label_smoothing > 0:
            # Label smoothing: target 不是严格的 1/0
            losses = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            )
        else:
            losses = -F.logsigmoid(logits)

        loss = losses.mean()

        # ── 指标 ──────────────────────────────────────
        with torch.no_grad():
            # Accuracy: chosen 的 ratio 是否 > rejected
            accuracy = (log_ratio_chosen > log_ratio_rejected).float().mean()

            # Reward (隐式): β * log_ratio
            chosen_rewards = (self.beta * log_ratio_chosen).mean()
            rejected_rewards = (self.beta * log_ratio_rejected).mean()
            reward_margin = chosen_rewards - rejected_rewards

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "chosen_rewards": chosen_rewards.item(),
            "rejected_rewards": rejected_rewards.item(),
            "reward_margin": reward_margin.item(),
            "logits_mean": logits.mean().item(),
        }


def compute_log_probs(
    model,
    input_ids: torch.Tensor,        # (B, seq_len)
    attention_mask: torch.Tensor,   # (B, seq_len)
    labels: torch.Tensor,           # (B, seq_len) — 只计算 label=-100 以外位置的 log-prob
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算模型在给定 labels 上的 token-level log-probabilities。

    返回：
        (B,) 每个样本的平均 token log-prob（只对 labels != -100 的位置）
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    logits = outputs.logits  # (B, seq_len, vocab)

    # Shift: 预测位置 t 对应 label 位置 t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Token-level cross-entropy（不做 reduction）
    token_log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather：取出每个位置上正确 token 的 log-prob
    per_token_log_probs = token_log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, seq_len-1)

    # 只对非 padding 位置求平均
    mask = (shift_labels != -100).float()
    sum_log_probs = (per_token_log_probs * mask).sum(dim=1)
    n_tokens = mask.sum(dim=1).clamp(min=1)

    return sum_log_probs / n_tokens  # (B,)
