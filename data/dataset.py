"""
遥感图像描述数据集加载器。

支持的数据集：
  - RSICD（10,921 张图像，每张 5 句描述，30+ 类别）
  - UCM-Captions（2,100 张图像，每张 5 句描述，21 类别）
  - Sydney-Captions（613 张图像，每张 5 句描述，7 类别）

数据格式（JSONL）：
  {"image": "路径/到/图片.jpg", "captions": ["...", "..."], "category": "机场"}

典型用法：
    >>> dataset = RemoteSensingCaptionDataset(
    ...     data_paths="data/processed/train.jsonl",
    ...     split="train",
    ...     transform=transforms_pipeline,
    ... )
    >>> sample = dataset[0]
    >>> print(sample.keys())  # dict_keys(['pixel_values', 'captions', 'category', 'image_path'])
"""

import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class RemoteSensingCaptionDataset:
    """多源遥感图像描述数据集加载器。

    支持跨数据集加载、训练/验证/测试划分、以及可选的类别均衡采样。

    关键约定：
      - 训练模式 (split="train")：__getitem__ 随机返回 1 条 caption
      - 评估模式 (split="val"/"test")：__getitem__ 返回全部 5 条 caption
      - 所有图像路径相对于 JSONL 文件所在目录解析
    """

    def __init__(
        self,
        data_paths: Union[str, List[str]],
        split: str = "train",
        transform: Optional[object] = None,
        max_length: int = 512,
        split_ratios: Optional[Dict[str, float]] = None,
        seed: int = 42,
        base_dir: Optional[str] = None,
    ):
        """
        参数：
            data_paths: JSONL 标注文件路径（单个或多个），多个时会合并。
            split: 数据集划分，取 "train" / "val" / "test" / "all"。
                   "all" 返回全部数据不分片。
            transform: Albumentations Compose 对象（可选）。
                       传入后 __getitem__ 返回 tensor 而非 PIL Image。
            max_length: 描述文本的最大 token 长度（保留字段，tokenize 在 collator 中执行）。
            split_ratios: 自定义划分比例，如 {"train": 0.8, "val": 0.1, "test": 0.1}。
                         为 None 时使用默认比例。
            seed: 随机种子，保证划分可复现。
            base_dir: 图像文件的基准目录。为 None 时自动使用 JSONL 文件所在目录。
        """
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        self.data_paths = [Path(p) for p in data_paths]
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.seed = seed
        self.base_dir = base_dir

        # 默认划分比例
        self.split_ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}

        # ── 加载全部样本 ────────────────────────
        self.all_samples: List[Dict] = []
        self._load_all()

        # ── 按 split 筛选 ────────────────────────
        if split == "all":
            self.samples = self.all_samples
        else:
            self.samples = self._split_samples()

        # ── 类别统计 ────────────────────────────
        self.category_counts = self._count_categories()

        logger.info(
            f"已加载 {len(self.samples)} 条样本（总样本 {len(self.all_samples)}），"
            f"split='{split}'，类别数={len(self.category_counts)}"
        )

    # ════════════════════════════════════════════════════════════════
    # 数据加载
    # ════════════════════════════════════════════════════════════════

    def _load_all(self):
        """加载并解析所有 JSONL 标注文件。"""
        for data_path in self.data_paths:
            if not data_path.exists():
                raise FileNotFoundError(f"标注文件不存在：{data_path}")

            # 确定图像基准目录
            base = self.base_dir or data_path.parent

            with open(data_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过 {data_path}:{line_num} —— JSON 解析错误：{e}")
                        continue

                    # ── 必需字段校验 ──────────────
                    if "image" not in sample:
                        logger.warning(f"跳过 {data_path}:{line_num} —— 缺少 'image' 字段")
                        continue
                    if "captions" not in sample or not sample["captions"]:
                        logger.warning(f"跳过 {data_path}:{line_num} —— 缺少 'captions' 字段")
                        continue

                    # ── 解析图像路径（相对 → 绝对） ──
                    image_path = Path(sample["image"])
                    if not image_path.is_absolute():
                        image_path = base / image_path

                    self.all_samples.append({
                        "image_path": str(image_path),
                        "captions": sample["captions"],
                        "category": sample.get("category", "unknown"),
                        "metadata": sample.get("metadata", {}),
                        "source": sample.get("metadata", {}).get("source", data_path.stem),
                    })

        logger.info(f"从 {len(self.data_paths)} 个文件加载了 {len(self.all_samples)} 条原始样本")

    # ════════════════════════════════════════════════════════════════
    # 数据集划分
    # ════════════════════════════════════════════════════════════════

    def _split_samples(self) -> List[Dict]:
        """按 split_ratios 分层划分，返回当前 split 对应的样本子集。

        分层策略：优先按 category 分层，保证各类别均匀分布在各子集中。
        """
        rng = random.Random(self.seed)

        # ── 按类别分组 ──────────────────────────
        by_category = defaultdict(list)
        for sample in self.all_samples:
            by_category[sample["category"]].append(sample)

        train_list, val_list, test_list = [], [], []

        for category, items in by_category.items():
            # 每个类别内部先打乱
            shuffled = list(items)
            rng.shuffle(shuffled)

            n = len(shuffled)
            n_train = max(1, int(n * self.split_ratios.get("train", 0.8)))
            n_val = max(1, int(n * self.split_ratios.get("val", 0.1))) if "val" in self.split_ratios else 0
            # 剩余归 test

            train_list.extend(shuffled[:n_train])

            if n_val > 0:
                val_list.extend(shuffled[n_train:n_train + n_val])
                test_list.extend(shuffled[n_train + n_val:])
            else:
                test_list.extend(shuffled[n_train:])

        # ── 再次打乱（跨类别混合） ──────────────
        rng.shuffle(train_list)
        rng.shuffle(val_list)
        rng.shuffle(test_list)

        logger.info(
            f"数据划分完成 —— train: {len(train_list)}, "
            f"val: {len(val_list)}, test: {len(test_list)}"
        )

        split_map = {"train": train_list, "val": val_list, "test": test_list}
        return split_map[self.split]

    def _count_categories(self) -> Dict[str, int]:
        """统计各类别样本数量。"""
        counts = defaultdict(int)
        for sample in self.samples:
            counts[sample["category"]] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # ════════════════════════════════════════════════════════════════
    # 样本获取
    # ════════════════════════════════════════════════════════════════

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """返回单条样本。

        训练模式：随机选择 1 条 caption（多 caption = 隐式数据增强）
        评估模式：返回全部 captions（用于多参考评估）

        返回字典包含：
            - pixel_values: PIL.Image 或 torch.Tensor（取决于是否传入 transform）
            - captions: str（训练）或 List[str]（评估）
            - category: str  地物类别标签
            - image_path: str 图像文件路径（用于错误分析）
        """
        sample = self.samples[idx]

        # ── 加载图像 ────────────────────────────
        image = Image.open(sample["image_path"]).convert("RGB")

        # ── 应用变换（如果有） ──────────────────
        if self.transform is not None:
            image_np = np.array(image)  # HWC uint8
            transformed = self.transform(image=image_np)
            pixel_values = transformed["image"]  # CHW float32 tensor 或 augmented HWC uint8
        else:
            pixel_values = image  # 保持 PIL Image，交给 collator 处理

        # ── 选择 caption ────────────────────────
        if self.split == "train":
            # 训练时随机选 1 条，增加多样性
            caption = random.choice(sample["captions"])
        else:
            # 评估时返回全部 captions
            caption = sample["captions"]

        return {
            "pixel_values": pixel_values,
            "captions": caption,
            "category": sample["category"],
            "image_path": sample["image_path"],
        }

    # ════════════════════════════════════════════════════════════════
    # 统计与分析
    # ════════════════════════════════════════════════════════════════

    def get_category_distribution(self) -> Dict[str, int]:
        """返回各类别样本数量统计（按样本数降序排列）。"""
        return self.category_counts

    def get_summary(self) -> Dict:
        """返回数据集的汇总统计信息。

        返回：
            {
                "total_samples": int,
                "num_categories": int,
                "category_distribution": Dict[str, int],
                "avg_captions_per_sample": float,
                "sources": List[str],
            }
        """
        total_captions = sum(
            len(s["captions"]) for s in self.samples
        )
        sources = list(set(s["source"] for s in self.samples))

        return {
            "total_samples": len(self.samples),
            "num_categories": len(self.category_counts),
            "category_distribution": self.category_counts,
            "avg_captions_per_sample": total_captions / max(len(self.samples), 1),
            "sources": sources,
        }

    def find_samples_by_category(self, category: str) -> List[int]:
        """返回指定类别的所有样本索引列表。"""
        return [
            i for i, s in enumerate(self.samples)
            if s["category"] == category
        ]

    def get_sample_with_all_references(self, idx: int) -> Dict:
        """返回包含全部 5 条参考描述的样本（用于评估时的详细对比）。"""
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform is not None:
            image_np = np.array(image)
            transformed = self.transform(image=image_np)
            pixel_values = transformed["image"]
        else:
            pixel_values = image

        return {
            "pixel_values": pixel_values,
            "captions": sample["captions"],  # 全部参考描述
            "category": sample["category"],
            "image_path": sample["image_path"],
        }
