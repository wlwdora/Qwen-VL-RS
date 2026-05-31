"""评估指标单元测试。"""

import pytest


class TestCaptioningMetrics:
    """测试各评估指标的计算正确性。"""

    def test_bleu_computation(self):
        """BLEU 指标计算应与参考实现一致。"""
        pass  # TODO: 实现测试

    def test_cider_computation(self):
        """CIDEr-D 指标计算应与 pycocoevalcap 一致。"""
        pass  # TODO: 实现测试

    def test_chair_metric(self):
        """CHAIR 幻觉指标应能正确检测不在图中的物体。"""
        pass  # TODO: 实现测试

    def test_empty_prediction_handling(self):
        """空预测应有合理的边界处理（不崩溃，得分为 0）。"""
        pass  # TODO: 实现测试
