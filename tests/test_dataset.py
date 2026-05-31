"""遥感数据集加载单元测试。

运行方式：
    pytest tests/test_dataset.py -v
    python -m pytest tests/test_dataset.py -v
"""

import json
import os
import tempfile

import pytest
import numpy as np
from PIL import Image


# ── 测试 fixtures ──────────────────────────────────

@pytest.fixture
def dummy_jsonl():
    """创建包含假数据和对应图像的临时 JSONL + 图像目录。

    数据集需要真实图像文件才能通过 __getitem__（PIL 会 open 图像）。
    """
    categories = ["airport", "mountain", "river", "beach", "parking"]
    img_dir = tempfile.mkdtemp()

    # 创建 20 个图像文件（足够分层划分）
    for i in range(20):
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        img.save(os.path.join(img_dir, f"img_{i}.jpg"))

    # 创建 20 条样本（每类别 4 条）
    samples = []
    for i in range(20):
        cat = categories[i % len(categories)]
        img_path = os.path.join(img_dir, f"img_{i}.jpg")
        samples.append({
            "image": img_path,
            "captions": [
                f"a beautiful {cat} scene {j}" for j in range(5)
            ],
            "category": cat,
            "metadata": {"idx": i},
        })

    jsonl_path = os.path.join(img_dir, "data.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    yield jsonl_path

    # cleanup
    for i in range(20):
        p = os.path.join(img_dir, f"img_{i}.jpg")
        if os.path.exists(p):
            os.unlink(p)
    if os.path.exists(jsonl_path):
        os.unlink(jsonl_path)
    os.rmdir(img_dir)


# ════════════════════════════════════════════════════════════

class TestRemoteSensingCaptionDataset:
    """测试数据集加载、划分和样本获取。"""

    def test_dataset_loads_correctly(self, dummy_jsonl):
        """数据集应能正常加载且结构符合预期。"""
        from data.dataset import RemoteSensingCaptionDataset

        ds = RemoteSensingCaptionDataset(
            data_paths=dummy_jsonl,
            split="train",
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            seed=42,
        )

        assert len(ds) > 0, "训练集不应为空"
        # 20样本 × 0.7 train ≈ 14，但分层划分会有舍入
        assert 8 <= len(ds) <= 16, f"训练集样本数不合理，实际 {len(ds)}"

    def test_train_val_split(self, dummy_jsonl):
        """训练集/验证集/测试集之间不应有重叠。"""
        from data.dataset import RemoteSensingCaptionDataset

        train = RemoteSensingCaptionDataset(
            data_paths=dummy_jsonl, split="train",
            split_ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=42,
        )
        val = RemoteSensingCaptionDataset(
            data_paths=dummy_jsonl, split="val",
            split_ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=42,
        )
        test = RemoteSensingCaptionDataset(
            data_paths=dummy_jsonl, split="test",
            split_ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=42,
        )

        # 提取各集合的样本路径检查重叠
        train_paths = set()
        for i in range(len(train)):
            train_paths.add(train[i]["image_path"])

        val_paths = set()
        for i in range(len(val)):
            val_paths.add(val[i]["image_path"])

        test_paths = set()
        for i in range(len(test)):
            test_paths.add(test[i]["image_path"])

        assert len(train_paths & val_paths) == 0, "训练集和验证集不应重叠"
        assert len(train_paths & test_paths) == 0, "训练集和测试集不应重叠"
        assert len(val_paths & test_paths) == 0, "验证集和测试集不应重叠"

        # 总样本数应等于原始样本数
        total = len(train) + len(val) + len(test)
        assert total == 20, f"总样本数应为 20，实际 {total}"

    def test_getitem_returns_correct_format(self, dummy_jsonl):
        """__getitem__ 应返回包含正确字段的字典。"""
        from data.dataset import RemoteSensingCaptionDataset

        ds = RemoteSensingCaptionDataset(
            data_paths=dummy_jsonl, split="train",
        )

        sample = ds[0]
        assert "pixel_values" in sample, "返回字典应包含 pixel_values"
        assert "captions" in sample, "返回字典应包含 captions"
        assert "category" in sample, "返回字典应包含 category"
        assert "image_path" in sample, "返回字典应包含 image_path"

        # pixel_values 应为 PIL Image 或 torch.Tensor
        pv = sample["pixel_values"]
        assert hasattr(pv, "convert") or hasattr(pv, "shape"), \
            "pixel_values 应为 PIL Image 或 Tensor"

        # 训练模式返回 1 条 caption
        assert isinstance(sample["captions"], str), \
            "训练模式下 captions 应为挑出的单个字符串"

    def test_category_distribution_is_balanced(self, dummy_jsonl):
        """类别分布应被正确统计和记录。"""
        from data.dataset import RemoteSensingCaptionDataset

        ds = RemoteSensingCaptionDataset(
            data_paths=dummy_jsonl, split="train",
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        )

        counts = ds.category_counts
        assert isinstance(counts, dict), "category_counts 应为字典"
        assert len(counts) > 0, "至少应有 1 个类别"
        assert all(isinstance(v, int) for v in counts.values()), \
            "所有计数应为整数"

    def test_get_sample_with_all_references(self, dummy_jsonl):
        """get_sample_with_all_references 应返回全部 5 条参考 caption。"""
        from data.dataset import RemoteSensingCaptionDataset

        ds = RemoteSensingCaptionDataset(
            data_paths=dummy_jsonl, split="test",
            split_ratios={"train": 0.4, "val": 0.3, "test": 0.3},
            seed=42,
        )

        assert len(ds) > 0, f"测试集不应为空，实际 {len(ds)}"
        sample = ds.get_sample_with_all_references(0)
        assert len(sample["captions"]) == 5, \
            f"应返回 5 条参考 caption，实际 {len(sample['captions'])}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header"])
