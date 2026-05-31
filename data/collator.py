"""多模态 Data Collator —— 视觉-语言训练的核心组件。

负责处理：
  - 图像批次的预处理（通过 AutoProcessor）
  - 文本变长序列的填充（input_ids / attention_mask / labels）
  - Qwen-VL "图像+文本" chat template 的构建
  - labels 掩码：只对 assistant 回复部分计算损失，prompt 部分被遮蔽

这是多模态训练中最容易出"静默 bug"的地方：
填充逻辑稍有偏差就会导致训练 loss 正常但推理结果完全错误。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image

# Qwen3 对话模板的特殊 token
QWEN_IMAGE_START = "<|vision_start|>"
QWEN_IMAGE_END = "<|vision_end|>"
ASSISTANT_START = "<|im_start|>assistant\n"
ASSISTANT_END = "<|im_end|>"


@dataclass
class MultiModalDataCollator:
    """将 (图像, 文本) 样本对整理成训练批次。

    参数：
        tokenizer: 分词器实例。
        processor: AutoProcessor 实例（负责图像预处理 + chat template）。
        max_length: 序列最大长度。
        padding: 是否进行填充。
        prompt_template: 使用的 prompt 模板名称，对应 data/prompts.py 中的 PROMPT_REGISTRY。
    """

    tokenizer: Any
    processor: Any
    max_length: int = 512
    padding: bool = True
    prompt: str = "请详细描述这张遥感图像的内容。"
    image_size: int = 512

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """将一个 batch 的样本整理成可直接送入模型的张量。

        参数：
            features: 样本字典列表，每个字典包含：
                      - "pixel_values": PIL.Image 或 torch.Tensor
                      - "captions": str（训练）或 List[str]（评估）
                      - "category": str
                      - "image_path": str

        返回：
            填充后的批次张量字典，包含：
                - input_ids, attention_mask, labels
                - pixel_values, image_grid_thw
        """
        batch_images = []
        batch_captions = []
        for f in features:
            batch_images.append(f["pixel_values"])
            # 训练时 captions 是单个 str，评估时是 List[str]
            caption = f["captions"]
            if isinstance(caption, list):
                # 评估模式：取第一条作为生成目标（实际评估在 eval 模块中用全部参考）
                caption = caption[0]
            batch_captions.append(caption)

        # ── 方式 1：如果图像已是 tensor，手动堆叠 + tokenize ──
        # if isinstance(batch_images[0], torch.Tensor):
        #     return self._collate_tensors(batch_images, batch_captions)

        # ── 方式 2：图像是 PIL Image，交给 processor 统一处理 ──
        return self._collate_pil(batch_images, batch_captions)

    def _collate_pil(
        self, images: List[Image.Image], captions: List[str]
    ) -> Dict[str, torch.Tensor]:
        """处理 PIL 图像批次：通过 processor 构建 chat template 并编码。"""
        # ── Step 1: 为每个样本构建对话 ─────────────────
        full_texts = []
        prompt_texts = []  # 仅 user prompt（不含 assistant 回复）

        for image, caption in zip(images, captions):
            # 完整对话（user + assistant）
            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": caption}],
                },
            ]
            full_text = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            full_texts.append(full_text)

            # 仅 user prompt（用于确定 labels 的分界点）
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt},
                    ],
                },
            ]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(prompt_text)

        # ── Step 2: 用 processor 处理 prompt 获取真实 prompt 长度 ──
        # 只算一次：同 batch 内所有样本 prompt 相同、图像尺寸相同，
        # prompt_length 完全一致，无需每样本重复计算。
        prompt_inputs = self.processor(
            text=[prompt_texts[0]],
            images=[images[0]],
            return_tensors="pt",
        )
        _prompt_len = prompt_inputs.input_ids.shape[1]

        # ── Step 3: 批量处理完整对话 ────────────────
        batch_inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )

        # ── Step 4: 创建 labels ──────────────────────
        input_ids = batch_inputs["input_ids"]  # (B, max_seq_len)
        labels = input_ids.clone()

        for i in range(input_ids.shape[0]):
            actual_prompt_len = min(_prompt_len, input_ids.shape[1])
            labels[i, :actual_prompt_len] = -100
            if "attention_mask" in batch_inputs:
                labels[i, batch_inputs["attention_mask"][i] == 0] = -100

        batch_inputs["labels"] = labels
        return batch_inputs

    def _collate_tensors(
        self, images: List[torch.Tensor], captions: List[str]
    ) -> Dict[str, torch.Tensor]:
        """处理预处理的 tensor 图像批次（图像已过 transforms 管线）。

        此时 processor.image_processor 被跳过，只使用 tokenizer 处理文本。
        """
        # ── 堆叠图像张量 ─────────────────────────
        pixel_values = torch.stack(images, dim=0)  # (B, 3, H, W)

        # ── 构建对话文本 ─────────────────────────
        full_texts = []
        prompt_texts = []

        for caption in captions:
            # 图像已预处理，chat template 中用占位文本表示图像
            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file.jpg"},  # processor 需要 image key
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": caption}],
                },
            ]
            full_text = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            full_texts.append(full_text)

            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file.jpg"},
                        {"type": "text", "text": self.prompt},
                    ],
                },
            ]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(prompt_text)

        # ── 计算 prompt 长度 ─────────────────────
        prompt_lengths = []
        for pt in prompt_texts:
            t = self.tokenizer(pt, return_tensors="pt")
            prompt_lengths.append(t.input_ids.shape[1])

        # ── 仅 tokenize 文本（图像已是 tensor，不经过 processor） ──
        text_inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )

        # ── 创建 labels ──────────────────────────
        input_ids = text_inputs["input_ids"]
        labels = input_ids.clone()

        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100
            if "attention_mask" in text_inputs:
                labels[i, text_inputs["attention_mask"][i] == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": text_inputs["attention_mask"],
            "labels": labels,
            "pixel_values": pixel_values,
        }


def build_collator(
    processor: Any,
    prompt_template: str = "请详细描述这张遥感图像的内容。",
    max_length: int = 512,
    image_size: int = 512,
) -> MultiModalDataCollator:
    """便捷工厂函数——一行创建 MultiModalDataCollator。

    用法：
        >>> from transformers import AutoProcessor
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        >>> collator = build_collator(processor, prompt="请详细描述这张遥感图像的内容。")
    """
    return MultiModalDataCollator(
        tokenizer=processor.tokenizer,
        processor=processor,
        max_length=max_length,
        padding=True,
        prompt=prompt_template,
        image_size=image_size,
    )
