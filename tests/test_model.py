"""模型前向传播和 LoRA 注入单元测试。

注意：这些测试需要 GPU 和已下载的基座模型，在没有 GPU 或模型时自动跳过。

运行方式：
    pytest tests/test_model.py -v
    python -m pytest tests/test_model.py -v

可通过环境变量覆盖模型路径：
    QWEN_MODEL_PATH=D:/Qwen/Qwen3-VL-2B-Instruct pytest tests/test_model.py -v
"""

import os

import pytest
import torch


# ── 辅助函数 ──────────────────────────────────────

def _model_path():
    """获取基座模型路径（环境变量或默认路径）。"""
    return os.environ.get(
        "QWEN_MODEL_PATH",
        "D:/Qwen/Qwen3-VL-2B-Instruct",
    )


def _adapter_path():
    """获取训练好的 LoRA adapter 路径。"""
    return os.environ.get(
        "QWEN_ADAPTER_PATH",
        "D:/work/Qwen-VL-RS/output/qwen_vl_rs_lora/best_model",
    )


def _has_model():
    """检查基座模型是否可用。"""
    return os.path.exists(_model_path())


def _has_gpu():
    """检查是否有可用的 CUDA GPU。"""
    return torch.cuda.is_available()


def _has_adapter():
    """检查是否已训练 LoRA adapter。"""
    return os.path.exists(_adapter_path())


# ── 跳过条件 ──────────────────────────────────────

requires_gpu = pytest.mark.skipif(
    not _has_gpu(), reason="需要 CUDA GPU"
)
requires_model = pytest.mark.skipif(
    not _has_model(), reason=f"基座模型未找到: {_model_path()}"
)
requires_adapter = pytest.mark.skipif(
    not _has_adapter(), reason=f"LoRA adapter 未找到: {_adapter_path()}"
)


# ════════════════════════════════════════════════════════════

class TestQwenVLForRemoteSensing:
    """测试模型加载和 LoRA adapter 注入。"""

    @requires_gpu
    @requires_model
    def test_model_loads_without_error(self):
        """模型应能从 checkpoint 正常加载。"""
        from models.qwen_vl_rs import QwenVLForRemoteSensing

        wrapper = QwenVLForRemoteSensing(
            model_path=_model_path(),
            torch_dtype=torch.float16,
        )
        wrapper.load()

        assert wrapper.model is not None, "基座模型不应为 None"
        assert wrapper.processor is not None, "处理器不应为 None"
        assert wrapper.tokenizer is not None, "分词器不应为 None"

        # 验证模型对象类型
        from transformers import Qwen3VLForConditionalGeneration
        assert isinstance(wrapper.model, Qwen3VLForConditionalGeneration), \
            f"模型类型应为 Qwen3VLForConditionalGeneration，实际 {type(wrapper.model)}"

    @requires_gpu
    @requires_model
    def test_lora_injection(self):
        """LoRA adapter 注入后，可训练参数数量应符合预期。"""
        from models.qwen_vl_rs import QwenVLForRemoteSensing

        wrapper = QwenVLForRemoteSensing(
            model_path=_model_path(),
            lora_config={"rank": 8, "alpha": 16, "dropout": 0.05},
            torch_dtype=torch.float16,
        )
        wrapper.load()

        assert wrapper.peft_model is not None, "LoRA 注入后 peft_model 不应为 None"

        # 可训练参数数量应在合理范围（几百万 ~ 一千万）
        trainable, total = wrapper.print_trainable_parameters()
        assert 1_000_000 <= trainable <= 20_000_000, \
            f"可训练参数 {trainable:,} 不在预期范围 [1M, 20M]"
        assert trainable < total * 0.1, \
            "可训练参数应占总参数 < 10%"

    @requires_gpu
    @requires_model
    def test_forward_pass_shape(self):
        """前向传播的输出 logits 形状应正确。"""
        from models.qwen_vl_rs import QwenVLForRemoteSensing

        wrapper = QwenVLForRemoteSensing(
            model_path=_model_path(),
            lora_config={"rank": 8, "alpha": 16},
            torch_dtype=torch.float16,
        )
        wrapper.load()
        model = wrapper.peft_model
        model.eval()

        # 构造假输入（最小 batch）
        device = next(model.parameters()).device
        batch_size, seq_len = 1, 32
        vocab_size = 151936  # Qwen3-VL 词表大小

        # Qwen3-VL forward 需要 input_ids + attention_mask
        # 不带图像的纯文本 forward
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert hasattr(outputs, "logits"), "输出应包含 logits"
        assert outputs.logits.shape == (batch_size, seq_len, vocab_size), \
            f"logits shape 应为 ({batch_size}, {seq_len}, {vocab_size})，" \
            f"实际 {outputs.logits.shape}"

    @requires_gpu
    @requires_model
    @requires_adapter
    def test_generation_output_format(self):
        """生成的描述文本应为非空字符串。"""
        from PIL import Image
        import numpy as np
        from models.qwen_vl_rs import QwenVLForRemoteSensing

        wrapper = QwenVLForRemoteSensing.from_pretrained(
            model_path=_model_path(),
            lora_adapter_path=_adapter_path(),
            torch_dtype=torch.float16,
        )
        model = wrapper.peft_model
        model.eval()

        # 构造一张假遥感图像 (RGB)
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        ).convert("RGB")

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_img},
                    {"type": "text", "text": "简要描述这张遥感图像"},
                ],
            },
        ]
        text = wrapper.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = wrapper.processor(
            text=[text], images=[dummy_img], return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = wrapper.generate(
                inputs, max_new_tokens=64, temperature=0.7
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, prompt_len:]
        caption = wrapper.decode(new_tokens, skip_special_tokens=True)

        assert isinstance(caption, str), f"caption 应为字符串，实际 {type(caption)}"
        assert len(caption.strip()) > 0, "生成的 caption 不应为空"

    @requires_gpu
    @requires_model
    def test_decode_1d_tensor(self):
        """decode() 应正确处理 1D tensor。"""
        from models.qwen_vl_rs import QwenVLForRemoteSensing

        wrapper = QwenVLForRemoteSensing(
            model_path=_model_path(),
            torch_dtype=torch.float16,
        )
        wrapper.load()

        # 测试 1D tensor
        token_ids = torch.tensor([1, 2, 3, 4, 5])  # 假的 token id
        result = wrapper.decode(token_ids)
        assert isinstance(result, str), f"1D 输入应返回 str，实际 {type(result)}"

        # 测试 2D tensor
        token_ids_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result_2d = wrapper.decode(token_ids_2d)
        assert isinstance(result_2d, list), f"2D 输入应返回 list，实际 {type(result_2d)}"
        assert len(result_2d) == 2, f"应返回 2 个字符串"


class TestDataPipeline:
    """集成测试：数据集 → Collator → 模型前向传播。"""

    @requires_gpu
    @requires_model
    def test_collator_to_model_batch(self):
        """完整管线：Collator 输出可直接输入模型。"""
        import tempfile, json
        import numpy as np
        from PIL import Image

        from models.qwen_vl_rs import QwenVLForRemoteSensing
        from data.collator import MultiModalDataCollator
        from data.dataset import RemoteSensingCaptionDataset

        # ── 创建临时 JSONL ─────────────────
        samples = []
        for i in range(3):
            img_path = os.path.join(tempfile.gettempdir(), f"test_img_{i}.jpg")
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            img.save(img_path)

            samples.append({
                "image": img_path,
                "captions": [f"test caption {j}" for j in range(5)],
                "category": "test",
            })

        jsonl_path = os.path.join(tempfile.gettempdir(), "test_data.jsonl")
        with open(jsonl_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        # ── 加载模型 ───────────────────────
        wrapper = QwenVLForRemoteSensing(
            model_path=_model_path(),
            torch_dtype=torch.float16,
        )
        wrapper.load()

        # ── 数据集 ──────────────────────────
        ds = RemoteSensingCaptionDataset(
            data_paths=jsonl_path, split="train",
        )

        # ── Collator ────────────────────────
        collator = MultiModalDataCollator(
            tokenizer=wrapper.tokenizer,
            processor=wrapper.processor,
            max_length=128,
            prompt="描述这张图",
        )

        # ── 构建批次 ────────────────────────
        batch_samples = [ds[i] for i in range(min(3, len(ds)))]
        batch = collator(batch_samples)

        assert "input_ids" in batch, "collator 输出应包含 input_ids"
        assert "labels" in batch, "collator 输出应包含 labels"
        assert batch["input_ids"].shape[0] == len(batch_samples), \
            "input_ids batch size 不匹配"

        # ── 前向传播 ────────────────────────
        device = next(wrapper.peft_model.parameters()).device if wrapper.peft_model else "cuda"
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = wrapper.peft_model(**inputs)

        assert hasattr(outputs, "logits"), "输出应包含 logits"
        assert not torch.isnan(outputs.logits).any(), "logits 中不应有 NaN"

        # ── 清理 ────────────────────────────
        os.unlink(jsonl_path)
        for i in range(3):
            p = os.path.join(tempfile.gettempdir(), f"test_img_{i}.jpg")
            if os.path.exists(p):
                os.unlink(p)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header"])
