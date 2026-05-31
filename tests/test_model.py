"""模型前向传播和 LoRA 注入单元测试。"""

import pytest


class TestQwenVLForRemoteSensing:
    """测试模型加载和 LoRA adapter 注入。"""

    def test_model_loads_without_error(self):
        """模型应能从 checkpoint 正常加载。"""
        pass  # TODO: 实现测试

    def test_lora_injection(self):
        """LoRA adapter 注入后，可训练参数数量应符合预期。"""
        pass  # TODO: 实现测试

    def test_forward_pass_shape(self):
        """前向传播的输出 logits 形状应正确。"""
        pass  # TODO: 实现测试

    def test_generation_output_format(self):
        """生成的描述文本应为非空字符串。"""
        pass  # TODO: 实现测试

    def test_merge_lora(self):
        """LoRA 合并后的模型应与训练态模型输出一致。"""
        pass  # TODO: 实现测试
