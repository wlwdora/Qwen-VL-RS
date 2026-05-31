"""单张图像和批量推理接口。

支持：
  - 单图推理（路径 / PIL Image / numpy 数组）
  - 批量推理
  - 多 prompt 对比（标准 vs 遥感专家 vs 思维链）
  - 自定义生成参数（温度、top_p、max_tokens）

用法：
    # 命令行
    python -m inference.infer --image path/to/image.jpg --checkpoint output/lora_adapter

    # 程序化
    >>> engine = InferenceEngine(checkpoint_path="output/lora_adapter")
    >>> engine.load()
    >>> caption = engine.predict("path/to/image.jpg")
    >>> print(caption)
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image

from data.prompts import PROMPT_REGISTRY, REMOTE_SENSING_PROMPTS

logger = logging.getLogger(__name__)


class InferenceEngine:
    """加载训练好的模型并执行推理。

    使用方式：
        >>> engine = InferenceEngine("output/lora_adapter")
        >>> engine.load()
        >>> result = engine.predict("farmland_001.jpg")
        >>> print(result)
    """

    def __init__(
        self,
        checkpoint_path: str,
        base_model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        参数：
            checkpoint_path: 训练好的模型 checkpoint 路径（完整模型或 LoRA adapter）。
            base_model_path: 基座模型路径（仅当 checkpoint 为 LoRA adapter 时需要）。
            device: "cuda" 或 "cpu"。
        """
        self.checkpoint_path = checkpoint_path
        self.base_model_path = base_model_path or self._infer_base_model()
        self.device = device

        # 延迟导入，避免 torch 不可用时崩溃
        self.model_wrapper = None
        self._loaded = False

    def _infer_base_model(self) -> str:
        """推断基座模型路径。"""
        for p in [
            "D:/Qwen/Qwen3-VL-2B-Instruct",
            "Qwen/Qwen3-VL-2B-Instruct",
        ]:
            if os.path.exists(p) or p.startswith("Qwen"):
                return p
        return "Qwen/Qwen3-VL-2B-Instruct"

    def load(self):
        """从 checkpoint 加载模型和处理器。"""
        from models.qwen_vl_rs import QwenVLForRemoteSensing

        logger.info(f"正在加载推理引擎...")
        t0 = time.time()

        self.model_wrapper = QwenVLForRemoteSensing.from_pretrained(
            model_path=self.base_model_path,
            lora_adapter_path=self.checkpoint_path
            if os.path.isdir(self.checkpoint_path)
            else None,
        )

        self._loaded = True
        logger.info(f"推理引擎加载完成，耗时 {time.time() - t0:.1f} 秒")

    def _ensure_loaded(self):
        """确保模型已加载。"""
        if not self._loaded:
            self.load()

    # ════════════════════════════════════════════════════════════
    # 单图推理
    # ════════════════════════════════════════════════════════════

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        verbose: bool = False,
    ) -> str:
        """为单张图像生成描述。

        参数：
            image: 图像路径、PIL Image 对象或 numpy 数组（HWC, uint8）。
            prompt: 自定义指令文本。为 None 时使用默认遥感专家 prompt。
            max_new_tokens: 最大生成长度。
            temperature: 采样温度（0.0 = 贪心）。
            do_sample: 是否采样。
            verbose: 是否打印耗时信息。

        返回：
            生成的描述文本。
        """
        self._ensure_loaded()

        # ── 加载图像 ────────────────────────
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(f"不支持的图像类型：{type(image)}")

        # ── 选择 prompt ─────────────────────
        if prompt is None:
            prompt = REMOTE_SENSING_PROMPTS[0]

        # ── 通过 processor 构建输入 ──────────
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = self.model_wrapper.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.model_wrapper.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
        ).to(self.device)

        # ── 生成 ────────────────────────────
        t0 = time.time()
        output_ids = self.model_wrapper.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        # ── 解码（只保留新生成的 token）─────
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, prompt_len:]
        caption = self.model_wrapper.decode(new_tokens, skip_special_tokens=True)

        if verbose:
            elapsed = time.time() - t0
            logger.info(f"生成耗时 {elapsed:.2f}s，产出 {len(new_tokens)} tokens")

        return caption.strip()

    # ════════════════════════════════════════════════════════════
    # 批量推理
    # ════════════════════════════════════════════════════════════

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Image.Image]],
        prompts: Optional[List[str]] = None,
        batch_size: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[str]:
        """为一个批次的图像生成描述。

        参数：
            images: 图像路径或 PIL Image 对象列表。
            prompts: 每张图对应的指令文本（长度须与 images 一致）。
                     为 None 时所有图像使用默认 prompt。
            batch_size: 推理批次大小。
            max_new_tokens: 最大生成长度。
            temperature: 采样温度。

        返回：
            描述文本列表，与输入顺序一致。
        """
        self._ensure_loaded()

        if prompts is None:
            prompts = [REMOTE_SENSING_PROMPTS[0]] * len(images)
        assert len(prompts) == len(images), "prompts 与 images 长度不一致"

        results = []
        n = len(images)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_images = images[start:end]
            batch_prompts = prompts[start:end]

            # ── 加载并处理批次图像 ─────────
            pil_images = []
            for img in batch_images:
                if isinstance(img, str):
                    pil_images.append(Image.open(img).convert("RGB"))
                elif isinstance(img, Image.Image):
                    pil_images.append(img.convert("RGB"))
                else:
                    raise TypeError(f"不支持的图像类型：{type(img)}")

            # ── 逐张构建输入并生成 ────────
            for pil_img, prompt in zip(pil_images, batch_prompts):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                text = self.model_wrapper.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = self.model_wrapper.processor(
                    text=[text],
                    images=[pil_img],
                    return_tensors="pt",
                ).to(self.device)

                output_ids = self.model_wrapper.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

                prompt_len = inputs["input_ids"].shape[1]
                new_tokens = output_ids[0, prompt_len:]
                caption = self.model_wrapper.decode(
                    new_tokens, skip_special_tokens=True
                )
                results.append(caption.strip())

        return results

    # ════════════════════════════════════════════════════════════
    # Prompt 对比
    # ════════════════════════════════════════════════════════════

    @torch.no_grad()
    def compare_prompts(
        self,
        image: Union[str, Image.Image],
        prompt_types: Optional[List[str]] = None,
        **generate_kwargs,
    ) -> Dict[str, str]:
        """在同一张图上对比不同 prompt 的生成效果。

        参数：
            image: 输入图像。
            prompt_types: 要对比的 prompt 类型列表。
                          可选：standard, remote_sensing, cot
                          默认全选。

        返回：
            {prompt_type: generated_caption} 的字典。

        用法（在 Gradio 中可作为对比面板的数据源）：
            >>> result = engine.compare_prompts("test.jpg")
            >>> for ptype, caption in result.items():
            ...     print(f"[{ptype}] {caption}")
        """
        if prompt_types is None:
            prompt_types = list(PROMPT_REGISTRY.keys())

        results = {}
        for ptype in prompt_types:
            prompts = PROMPT_REGISTRY.get(ptype, PROMPT_REGISTRY["standard"])
            # 每个类型取第一条 prompt（也可多条取平均，但通常选第一条）
            prompt = prompts[0] if isinstance(prompts, list) else prompts
            caption = self.predict(image, prompt=prompt, **generate_kwargs)
            results[ptype] = caption

        return results


# ════════════════════════════════════════════════════════════════
# CLI 入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen-VL-RS 推理")
    parser.add_argument("--image", type=str, required=True,
                        help="输入图像路径")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--base_model", type=str, default=None,
                        help="基座模型路径")
    parser.add_argument("--prompt", type=str, default=None,
                        help="自定义 prompt（默认使用遥感专家模板）")
    parser.add_argument("--prompt_type", type=str, default="remote_sensing",
                        choices=["standard", "remote_sensing", "cot"],
                        help="选择预设 prompt 类型")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--compare_prompts", action="store_true",
                        help="对比所有 prompt 类型")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出文件路径")

    args = parser.parse_args()

    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        base_model_path=args.base_model,
    )
    engine.load()

    if args.compare_prompts:
        results = engine.compare_prompts(
            args.image,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print("\n=== Prompt 对比结果 ===")
        for ptype, caption in results.items():
            print(f"\n[{ptype}]")
            print(f"  {caption}")
    else:
        prompt = args.prompt
        if prompt is None:
            prompts = PROMPT_REGISTRY.get(args.prompt_type, REMOTE_SENSING_PROMPTS)
            prompt = prompts[0] if isinstance(prompts, list) else prompts

        caption = engine.predict(
            args.image,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\n{caption}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(caption)
            print(f"\n结果已保存至：{args.output}")
