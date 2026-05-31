"""
Qwen3-VL 遥感图像描述模型封装 + LoRA 适配。

模型架构：
  视觉编码器 (Qwen3-VL ViT, 冻结)
       ↓
  MLP 投影层 (冻结或 LoRA)
       ↓
  大语言模型 (Qwen3, LoRA 微调)
       ↓
  文本输出 (描述 tokens)

关键设计决策：
  - 视觉编码器冻结：遥感图像的低级特征与自然图像共享，无需重训
  - LLM 部分用 LoRA 微调：领域词汇和描述风格需要适配，但不必全量训练
  - MLP 投影层可选训练：桥接视觉特征的领域差异

典型用法：
    >>> wrapper = QwenVLForRemoteSensing(
    ...     model_path="D:/Qwen/Qwen3-VL-2B-Instruct",
    ...     lora_config={"rank": 16, "alpha": 32, "dropout": 0.05},
    ... )
    >>> wrapper.load()
    >>> wrapper.print_trainable_parameters()
    >>> # 训练后
    >>> wrapper.merge_and_save("./output/merged_model")
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    ProcessorMixin,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

logger = logging.getLogger(__name__)


# ── 默认 LoRA 配置 ──────────────────────────────
# 这些值来自 configs/lora_config.yaml，可通过构造参数覆盖
DEFAULT_LORA_CONFIG = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",    # 注意力投影
        "gate_proj", "up_proj", "down_proj",         # FFN 投影（可选）
    ],
    "modules_to_save": None,  # 额外全量训练的模块，如 lm_head
}


@dataclass
class ModelInfo:
    """模型信息摘要。"""
    model_name: str
    total_params: int
    trainable_params: int
    trainable_ratio: float
    lora_config: Optional[Dict] = None


class QwenVLForRemoteSensing:
    """Qwen3-VL + LoRA 遥感描述模型封装类。

    封装了模型加载、LoRA 注入、训练/推理切换、合并导出等完整生命周期。
    """

    def __init__(
        self,
        model_path: str,
        lora_config: Optional[Dict] = None,
        torch_dtype: torch.dtype = torch.float16,  # CUDA 11.8 下 float16 更稳定
        device_map: str = "auto",
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = True,
    ):
        """
        参数：
            model_path: Qwen3-VL 模型 checkpoint 路径（HuggingFace ID 或本地路径）。
            lora_config: LoRA 参数字典，含 rank / alpha / dropout / target_modules。
                        为 None 时不注入 LoRA（仅加载基座模型）。
            torch_dtype: 模型权重的数值精度（bfloat16 推荐，节省显存）。
            device_map: 设备映射策略，"auto" 自动分配到可用 GPU。
            attn_implementation: 注意力实现方式（sdpa / flash_attention_2 / eager）。
            trust_remote_code: 是否信任远程代码（Qwen-VL 系列需要开启）。
        """
        self.model_path = model_path
        self.lora_config = lora_config or {}
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code

        # 运行时对象
        self.model: Optional[Qwen3VLForConditionalGeneration] = None
        self.peft_model: Optional[PeftModel] = None
        self.processor: Optional[ProcessorMixin] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        # 合并默认配置
        self._effective_lora = {**DEFAULT_LORA_CONFIG, **self.lora_config}

    # ════════════════════════════════════════════════════════════
    # 模型加载
    # ════════════════════════════════════════════════════════════

    def load(self):
        """加载基座模型、处理器和分词器。

        加载顺序：
          1. 加载 Qwen3VLForConditionalGeneration
          2. 加载 AutoProcessor（图像 + 文本预处理）
          3. 提取 tokenizer（便捷引用）
          4. 如果配置了 lora_config，注入 LoRA adapter
        """
        logger.info(f"正在加载模型：{self.model_path}")

        # ── 加载模型 ────────────────────────────
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.torch_dtype,  # 注意：新版 transformers 用 dtype，非 torch_dtype
            device_map=self.device_map,
            attn_implementation=self.attn_implementation,
            trust_remote_code=self.trust_remote_code,
        )

        # ── 加载处理器 ──────────────────────────
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer = self.processor.tokenizer

        # ── 可选：注入 LoRA ─────────────────────
        if self.lora_config:
            self.apply_lora()

        logger.info("模型加载完成")

    def _find_linear_modules(self) -> List[str]:
        """自动发现模型中所有 nn.Linear 层的名称。

        用于支持 target_modules="all-linear" 等便捷配置。
        """
        linear_modules = set()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 取最后一层名字（如 "model.layers.0.self_attn.q_proj" → "q_proj"）
                leaf_name = name.split(".")[-1]
                linear_modules.add(leaf_name)
        return sorted(linear_modules)

    def _resolve_target_modules(self) -> List[str]:
        """解析 target_modules 配置，处理 "all-linear" 等别名。"""
        target_modules = self._effective_lora.get("target_modules", [])
        if not target_modules:
            # 默认：只对注意力投影做 LoRA
            return ["q_proj", "k_proj", "v_proj", "o_proj"]

        # 展开 all-linear
        if "all-linear" in target_modules or "all_linear" in target_modules:
            return self._find_linear_modules()

        return target_modules

    # ════════════════════════════════════════════════════════════
    # LoRA 注入
    # ════════════════════════════════════════════════════════════

    def apply_lora(self) -> PeftModel:
        """为模型注入 LoRA adapter。

        步骤：
          1. 从 lora_config 字典构建 LoraConfig 对象
          2. 用 get_peft_model 包装模型
          3. 打印可训练参数统计
          4. 冻结视觉编码器（安全网：即使 target_modules 匹配也不微调 ViT）

        返回：
            注入 LoRA 后的 PeftModel 实例。
        """
        if self.model is None:
            raise RuntimeError("请先调用 load() 加载基座模型")

        if self.peft_model is not None:
            logger.warning("LoRA 已注入，跳过重复操作")
            return self.peft_model

        # ── 解析 target_modules ─────────────────
        target_modules = self._resolve_target_modules()

        # ── 构建 LoraConfig ─────────────────────
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self._effective_lora["rank"],
            lora_alpha=self._effective_lora["alpha"],
            lora_dropout=self._effective_lora["dropout"],
            target_modules=target_modules,
            modules_to_save=self._effective_lora.get("modules_to_save"),
            bias="none",
        )

        logger.info(
            f"注入 LoRA —— rank={peft_config.r}, alpha={peft_config.lora_alpha}, "
            f"dropout={peft_config.lora_dropout}, targets={target_modules}"
        )

        # ── 包装模型 ──────────────────────────
        self.peft_model = get_peft_model(self.model, peft_config)

        # ── 冻结视觉编码器（安全网） ──────────
        self._freeze_vision_encoder()

        # ── 打印统计 ──────────────────────────
        self.print_trainable_parameters()

        return self.peft_model

    def _freeze_vision_encoder(self):
        """确保视觉编码器的所有参数都被冻结。

        即使 PEFT 的 target_modules 意外匹配了 visual 模块的层名，
        此安全网也会强制冻结它们。
        """
        model = self.peft_model or self.model
        frozen_count = 0
        for name, param in model.named_parameters():
            if "visual" in name and param.requires_grad:
                param.requires_grad = False
                frozen_count += 1

        if frozen_count > 0:
            logger.info(f"安全网：额外冻结了 {frozen_count} 个视觉编码器参数")

    # ════════════════════════════════════════════════════════════
    # 参数统计
    # ════════════════════════════════════════════════════════════

    def get_model_info(self) -> ModelInfo:
        """返回模型信息摘要。"""
        model = self.peft_model or self.model
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return ModelInfo(
            model_name=self.model_path,
            total_params=total,
            trainable_params=trainable,
            trainable_ratio=trainable / total if total > 0 else 0.0,
            lora_config=self._effective_lora if self.peft_model else None,
        )

    def print_trainable_parameters(self):
        """打印可训练参数统计 —— 面试必备输出。

        输出示例：
            ==================================================
            trainable params:  12,845,056  (0.52% of total)
            total params:    2,471,338,496
            ==================================================
        """
        info = self.get_model_info()
        print("=" * 50)
        print(f"  trainable params:  {info.trainable_params:>12,}  "
              f"({info.trainable_ratio*100:.2f}% of total)")
        print(f"  total params:      {info.total_params:>12,}")
        print("=" * 50)

        # 逐模块细化
        model = self.peft_model or self.model
        module_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 提取顶层模块名
                top_module = name.split(".")[0]
                module_stats[top_module] = module_stats.get(top_module, 0) + param.numel()

        if module_stats:
            for module_name, count in sorted(module_stats.items(), key=lambda x: -x[1]):
                print(f"    {module_name}: {count:>12,}")

    # ════════════════════════════════════════════════════════════
    # 推理生成
    # ════════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """为一个批次的图像生成描述。

        参数：
            inputs: Collator 输出的批次字典，包含 input_ids、attention_mask、
                    pixel_values、image_grid_thw 等。
            max_new_tokens: 最大生成长度。
            temperature: 采样温度（越低越确定）。
            top_p: nucleus 采样阈值。
            do_sample: 是否进行采样（False 则贪心解码）。

        返回：
            (B, new_tokens) 形状的 token id 张量。
        """
        model = self.peft_model or self.model
        if model is None:
            raise RuntimeError("模型尚未加载，请先调用 load()")

        model.eval()

        # 构建 generate 输入
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id
            or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs,
        }

        # 将必要的输入送入模型
        generate_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
        }
        # 过滤掉 None 值
        generate_inputs = {k: v for k, v in generate_inputs.items() if v is not None}

        generate_inputs = {
            k: v.to(model.device) for k, v in generate_inputs.items()
        }

        outputs = model.generate(**generate_inputs, **generate_kwargs)
        return outputs

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """将 token id 解码为文本。

        参数：
            token_ids: (seq_len,) 或 (B, seq_len) 的 token id 张量。
            skip_special_tokens: 是否跳过特殊 token。

        返回：
            解码后的文本字符串（或字符串列表）。
            - 输入 (seq_len,) → 返回 str
            - 输入 (B, seq_len) → 返回 List[str]
        """
        if self.tokenizer is None:
            raise RuntimeError("分词器尚未加载")

        # batch_decode 要求 2D tensor (B, seq_len)，1D 会被误当作 batch 维度
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # (seq_len,) → (1, seq_len)
            result = self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
            return result[0]  # 返回 str
        else:
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)  # 返回 List[str]

    def predict(
        self,
        inputs: Dict[str, torch.Tensor],
        remove_prompt: bool = True,
        **generate_kwargs,
    ) -> List[str]:
        """快捷预测方法：输入 → 生成 → 解码 → 清理 → 返回文本列表。

        参数：
            inputs: Collator 输出的批次字典。
            remove_prompt: 是否自动去除 prompt 部分（仅保留生成的描述）。
            **generate_kwargs: 传递给 generate() 的参数。

        返回：
            描述文本列表。
        """
        output_ids = self.generate(inputs, **generate_kwargs)
        texts = self.decode(output_ids)

        if remove_prompt:
            # 去掉 input_ids 中已有的 prompt，只保留新生成的部分
            input_ids = inputs["input_ids"]
            prompt_lengths = input_ids.shape[1]
            output_tokens = output_ids[:, prompt_lengths:]
            texts = self.decode(output_tokens)

        return texts

    # ════════════════════════════════════════════════════════════
    # LoRA 合并与保存
    # ════════════════════════════════════════════════════════════

    def merge_and_save(self, output_path: str, push_to_hub: bool = False):
        """将 LoRA 权重合并到基座模型中并保存。

        合并后的模型是标准的 Qwen3VLForConditionalGeneration，
        不再依赖 PEFT 库，可直接用于推理部署。

        参数：
            output_path: 合并后模型的保存目录。
            push_to_hub: 是否推送到 HuggingFace Hub。

        步骤：
          1. 合并 LoRA 权重到基座模型
          2. 保存完整模型
          3. 保存处理器和分词器
          4. 可选的 HF Hub 推送
        """
        if self.peft_model is None:
            raise RuntimeError("没有 LoRA adapter，无需合并。请直接保存 model 即可。")

        logger.info(f"正在合并 LoRA 权重到基座模型...")
        merged_model = self.peft_model.merge_and_unload()
        logger.info("LoRA 权重已合并")

        # 保存合并后的模型
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(output_path)
        logger.info(f"模型已保存至：{output_path}")

        # 保存处理器
        if self.processor is not None:
            self.processor.save_pretrained(output_path)
            logger.info("处理器已保存")

        # 可选推送
        if push_to_hub:
            merged_model.push_to_hub(output_path)
            if self.processor is not None:
                self.processor.push_to_hub(output_path)

        return merged_model

    def save_lora(self, output_path: str):
        """仅保存 LoRA adapter 权重（不合并）。

        adapter 文件极小（~10MB），适合快速分享和版本管理。
        """
        if self.peft_model is None:
            raise RuntimeError("没有 LoRA adapter 可保存")

        os.makedirs(output_path, exist_ok=True)
        self.peft_model.save_pretrained(output_path)
        logger.info(f"LoRA adapter 已保存至：{output_path}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        lora_adapter_path: Optional[str] = None,
        **kwargs,
    ) -> "QwenVLForRemoteSensing":
        """加载已训练的模型（基座 + 可选 LoRA adapter）。

        用法：
            >>> wrapper = QwenVLForRemoteSensing.from_pretrained(
            ...     "D:/Qwen/Qwen3-VL-2B-Instruct",
            ...     lora_adapter_path="./output/lora_adapter",
            ... )
        """
        wrapper = cls(model_path=model_path, **kwargs)
        wrapper.load()

        if lora_adapter_path:
            # 加载已训练的 LoRA adapter
            from peft import PeftModel
            wrapper.peft_model = PeftModel.from_pretrained(
                wrapper.model, lora_adapter_path
            )
            logger.info(f"已加载 LoRA adapter：{lora_adapter_path}")

        return wrapper


# ════════════════════════════════════════════════════════════════
# 便捷函数
# ════════════════════════════════════════════════════════════════

def create_model(
    model_path: str,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    torch_dtype: str = "bfloat16",
) -> QwenVLForRemoteSensing:
    """一行创建带 LoRA 的模型。

    用法：
        >>> model = create_model("D:/Qwen/Qwen3-VL-2B-Instruct", lora_rank=16)
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    lora_config = {
        "rank": lora_rank,
        "alpha": lora_alpha,
        "dropout": lora_dropout,
    }
    if target_modules:
        lora_config["target_modules"] = target_modules

    wrapper = QwenVLForRemoteSensing(
        model_path=model_path,
        lora_config=lora_config,
        torch_dtype=dtype,
    )
    wrapper.load()
    return wrapper
