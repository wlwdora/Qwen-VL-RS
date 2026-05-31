"""遥感数据集加载单元测试。"""

import pytest


class TestRemoteSensingCaptionDataset:
    """测试数据集加载、划分和样本获取。"""

    def test_dataset_loads_correctly(self):
        """数据集应能正常加载且结构符合预期。"""
        pass  # TODO: 实现测试

    def test_train_val_split(self):
        """训练集/验证集/测试集之间不应有重叠。"""
        pass  # TODO: 实现测试

    def test_getitem_returns_correct_format(self):
        """__getitem__ 应返回包含 pixel_values / input_ids / labels 的字典。"""
        pass  # TODO: 实现测试

    def test_category_distribution_is_balanced(self):
        """类别分布应被正确统计和记录。"""
        pass  # TODO: 实现测试
