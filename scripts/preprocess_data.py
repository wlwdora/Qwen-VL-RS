"""数据集预处理脚本：将原始遥感数据集转换为项目标准 JSONL 格式。

支持的数据集格式：
  1. RSICD_optimal (GitHub: 201528014227051/RSICD_optimal)
     - dataset_rsicd.json  +  RSICD_images/
     - dataset_ucm.json    +  UCM_images/
     - dataset_sydney.json +  Sydney_images/

用法：
    # 预处理全部三个数据集
    python scripts/preprocess_data.py --all

    # 预处理单个数据集
    python scripts/preprocess_data.py --dataset rsicd

    # 指定自定义路径
    python scripts/preprocess_data.py --dataset rsicd \
        --annotation data/raw/RSICD_optimal/dataset_rsicd.json \
        --image_dir data/raw/RSICD_optimal/RSICD_images/

输入格式 (JSON):
    {
      "images": [
        {
          "filename": "001.jpg",
          "sentences": [{"raw": "caption1"}, {"raw": "caption2"}, ...],
          "split": "train" | "val" | "test",
          "category": "airport"  (可选)
        },
        ...
      ],
      "dataset": "rsicd"
    }

输出格式 (JSONL):
    {"image": "/abs/path/to/image.jpg", "captions": ["...", "..."], "category": "airport"}
"""

import json
import os
import argparse
import shutil
from pathlib import Path
from collections import defaultdict


# ── 数据集注册表 ──────────────────────────────
# 预设每个数据集在 RSICD_optimal 仓库中的默认路径
DATASET_REGISTRY = {
    "rsicd": {
        "annotation": "dataset_rsicd.json",
        "image_dir": "RSICD_images",
        "name": "RSICD",
    },
    "ucm": {
        "annotation": "dataset_ucm.json",
        "image_dir": "UCM_images",
        "name": "UCM-Captions",
    },
    "sydney": {
        "annotation": "dataset_sydney.json",
        "image_dir": "Sydney_images",
        "name": "Sydney-Captions",
    },
}


def load_annotations(annotation_path: str) -> list:
    """加载并解析标注 JSON 文件。

    支持的格式变体：
      - {"images": [...]}     RSICD_optimal 标准格式
      - [...]                 纯列表格式
      - {"annotations": [...]}
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 尝试不同 key
    if isinstance(data, list):
        samples = data
    elif "images" in data:
        samples = data["images"]
    elif "annotations" in data:
        samples = data["annotations"]
    else:
        # 最后一个兜底：取第一个列表类型的值
        for key, val in data.items():
            if isinstance(val, list):
                samples = val
                break
        else:
            raise ValueError(
                f"无法识别标注文件格式。"
                f"期望 {{\"images\": [...]}} 或 [...]，"
                f"实际顶层 key: {list(data.keys())}"
            )

    return samples


def extract_captions(sample: dict) -> list:
    """从单条标注中提取所有描述文本。

    支持的格式：
      - {"sentences": [{"raw": "text"}, ...]}
      - {"captions": ["text1", "text2", ...]}
      - {"sentences": ["text1", "text2", ...]}
    """
    # 格式 1: {"sentences": [{"raw": "..."}, ...]}
    if "sentences" in sample:
        sents = sample["sentences"]
        if isinstance(sents, list) and len(sents) > 0:
            if isinstance(sents[0], dict):
                return [s.get("raw", s.get("caption", "")) for s in sents]
            elif isinstance(sents[0], str):
                return sents

    # 格式 2: {"captions": [...]}
    if "captions" in sample and isinstance(sample["captions"], list):
        return sample["captions"]

    # 格式 3: 顶层单个 caption
    for key in ["caption", "description", "text"]:
        if key in sample:
            return [sample[key]]

    return []


def extract_category(sample: dict) -> str:
    """从标注中提取类别标签。

    策略：
      1. 优先取 category / label 等字段
      2. 其次从 filename 中推断（例如 airport_1.jpg → airport）
      3. 都没有则返回 "unknown"
    """
    for key in ["category", "label", "class", "scene", "type", "land_cover"]:
        if key in sample:
            return str(sample[key]).lower().strip()

    # 从文件名推断类别（除去数字和扩展名）
    filename = sample.get("filename", sample.get("image", ""))
    if filename:
        import re
        # 去掉路径和扩展名
        basename = os.path.basename(filename)
        stem = os.path.splitext(basename)[0]
        # 去掉尾部数字和特殊字符 → 类别名
        category = re.sub(r'[\d_\-]+$', '', stem).strip('_')
        if category:
            return category.lower()

    return "unknown"


def extract_split(sample: dict) -> str:
    """从标注中提取数据集划分信息。"""
    for key in ["split", "subset", "set", "partition"]:
        if key in sample:
            val = str(sample[key]).lower().strip()
            if val in ["train", "training"]:
                return "train"
            elif val in ["val", "validation", "dev"]:
                return "val"
            elif val in ["test", "testing"]:
                return "test"
            return val
    return "train"  # 默认


def preprocess_dataset(
    annotation_path: str,
    image_dir: str,
    output_path: str,
    dataset_name: str,
    splits: list = None,
) -> dict:
    """预处理单个数据集。

    参数：
        annotation_path: 标注 JSON 文件路径。
        image_dir: 图像目录路径。
        output_path: 输出 JSONL 文件路径。
        dataset_name: 数据集名称（用于 metadata）。
        splits: 要包含的 split 列表。None = 全部。

    返回：
        {"total": int, "by_category": dict, "by_split": dict}
    """
    print(f"\n{'='*50}")
    print(f"  预处理: {dataset_name}")
    print(f"  标注文件: {annotation_path}")
    print(f"  图像目录: {image_dir}")
    print(f"{'='*50}")

    if splits is None:
        splits = ["train", "val", "test"]

    # ── 加载标注 ──────────────────────────
    samples = load_annotations(annotation_path)
    print(f"  原始样本数: {len(samples)}")

    # ── 扫描图像目录 ──────────────────────
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")

    # 建立文件名→路径的映射
    image_files = {}
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]:
        for img_path in image_dir.glob(ext):
            image_files[img_path.name.lower()] = str(img_path.absolute())
        # 同时扫描一级子目录
        for img_path in image_dir.glob(f"*/{ext}"):
            image_files[img_path.name.lower()] = str(img_path.absolute())

    print(f"  图像文件数: {len(image_files)}")

    # ── 转换 ──────────────────────────────
    stats = {"total": 0, "by_category": defaultdict(int), "by_split": defaultdict(int)}
    skipped_no_image = 0
    skipped_no_captions = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            # 提取 split
            split = extract_split(sample)

            # 如果指定了 split 过滤
            if splits and split not in splits:
                continue

            # 提取图像路径
            filename = sample.get("filename", sample.get("image", sample.get("img", "")))
            # 处理多层级文件名
            if "/" in filename:
                filename = os.path.basename(filename)
            if "\\" in filename:
                filename = os.path.basename(filename)

            # 在图像目录中查找
            image_path = image_files.get(filename.lower())
            if image_path is None:
                skipped_no_image += 1
                continue

            # 提取描述
            captions = extract_captions(sample)
            if not captions:
                skipped_no_captions += 1
                continue

            # 提取类别
            category = extract_category(sample)

            # 构建标准格式
            record = {
                "image": image_path,
                "captions": captions,
                "category": category,
                "metadata": {
                    "source": dataset_name,
                    "split": split,
                    "original_filename": filename,
                },
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["total"] += 1
            stats["by_category"][category] += 1
            stats["by_split"][split] += 1

    print(f"  输出样本数: {stats['total']}")
    print(f"  类别数: {len(stats['by_category'])}")
    print(f"  Split 分布: {dict(stats['by_split'])}")
    if skipped_no_image:
        print(f"  [WARN] 未匹配到图像: {skipped_no_image}")
    if skipped_no_captions:
        print(f"  [WARN] 无描述文本: {skipped_no_captions}")
    print(f"  输出文件: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="将原始遥感数据集转换为 Qwen-VL-RS 标准 JSONL 格式"
    )
    parser.add_argument("--all", action="store_true",
                        help="预处理全部三个数据集")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["rsicd", "ucm", "sydney"],
                        help="预处理单个数据集")
    parser.add_argument("--annotation", type=str, default=None,
                        help="标注 JSON 文件路径（覆盖默认路径）")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="图像目录路径（覆盖默认路径）")
    parser.add_argument("--repo_dir", type=str,
                        default="data/raw/RSICD_optimal",
                        help="RSICD_optimal 仓库的本地路径")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="JSONL 输出目录")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="要包含的数据划分（逗号分隔）")
    args = parser.parse_args()

    # ── 确定要处理的数据集 ─────────────
    datasets_to_process = []
    if args.all:
        datasets_to_process = ["rsicd", "ucm", "sydney"]
    elif args.dataset:
        datasets_to_process = [args.dataset]
    else:
        parser.print_help()
        print("\n请指定 --dataset 或 --all")
        return

    repo_dir = Path(args.repo_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = args.splits.split(",")

    # ── 批量处理 ────────────────────────
    all_stats = {}
    for ds_name in datasets_to_process:
        ds_info = DATASET_REGISTRY[ds_name]

        # 路径：优先使用命令行参数，其次使用默认 repo 结构
        annotation_path = args.annotation or str(repo_dir / ds_info["annotation"])
        image_dir = args.image_dir or str(repo_dir / ds_info["image_dir"])
        output_path = output_dir / f"{ds_name}.jsonl"

        try:
            stats = preprocess_dataset(
                annotation_path=annotation_path,
                image_dir=image_dir,
                output_path=str(output_path),
                dataset_name=ds_info["name"],
                splits=splits,
            )
            all_stats[ds_name] = stats
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {ds_name}: {e}")
            continue
        except Exception as e:
            print(f"\n  [ERROR] {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── 汇总 ────────────────────────────
    print(f"\n{'='*60}")
    print(f"  预处理完成汇总")
    print(f"{'='*60}")
    total = 0
    for ds_name, stats in all_stats.items():
        print(f"  {ds_name:10s}: {stats['total']:>6d} 样本, "
              f"{len(stats['by_category']):>3d} 类别")
        total += stats["total"]
    print(f"  {'合计':10s}: {total:>6d} 样本")
    print(f"\n  输出目录: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
