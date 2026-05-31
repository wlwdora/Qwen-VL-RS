"""多模态 Data Collator —— 视觉-语言训练的核心组件。

负责处理：
  - 图像批次的堆叠（pixel_values）
  - 文本变长序列的填充（input_ids / attention_mask / labels）
  - Qwen-VL "图像+文本" 格式的 attention mask 构建

这是多模态训练中最容易出"静默 bug"的地方：
填充逻辑稍有偏差就会导致训练正确率正常但推理结果完全错误。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class MultiModalDataCollator:
    """将 (图像, 文本) 样本对整理成训练批次。

    参数：
        tokenizer: 分词器实例。
        processor: AutoProcessor 实例（负责图像预处理）。
        max_length: 序列最大长度。
        padding: 是否进行填充。
    """

    tokenizer: Any
    processor: Any
    max_length: int = 512
    padding: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """将一个 batch 的样本整理成可直接送入模型的张量。

        参数：
            features: 样本字典列表，每个字典包含 "pixel_values"、
                      "input_ids"、"attention_mask"、"labels"。

        返回：
            填充后的批次张量字典。
        """
        # TODO: 实现 collation 逻辑
        # 1. 堆叠 pixel_values（经过 transforms 后尺寸已统一）
        # 2. 将 input_ids 和 labels 填充到 batch 内最大长度
        # 3. 构建正确的 attention mask
        raise NotImplementedError("【待实现】Data Collator")
