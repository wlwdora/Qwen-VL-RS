"""生成假数据用于管线验证。

当真实数据集尚未下载时，用此脚本生成少量假数据来验证：
  - 数据加载流程（Dataset / Collator）
  - 模型前向传播 shape 检查
  - loss 计算无 NaN
  - 训练循环完整跑通

用法：
    python scripts/generate_dummy_data.py --num_samples 100 --output_dir data/processed/

生成物：
    data/raw/dummy/images/     — 随机彩色图像（224x224）
    data/processed/dummy.jsonl — 标准 JSONL 标注文件
"""

import json
import os
import random
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── 遥感地物类别 ─────────────────────────────
LAND_COVER_CATEGORIES = [
    "airport", "beach", "forest", "residential", "highway",
    "river", "farmland", "parking_lot", "mountain", "harbor",
    "desert", "grassland", "industrial", "lake", "bridge",
    "stadium", "orchard", "wetland", "snow", "coastline",
]

# ── 类别 → 描述模板 ──────────────────────────
CATEGORY_CAPTIONS = {
    "airport": [
        "An airport with {n} runways and several terminal buildings.",
        "Several airplanes parked near the airport terminals.",
        "An aerial view of an airport with hangars and {n} runways.",
    ],
    "beach": [
        "A sandy beach along the coastline with gentle waves.",
        "A beach area where the ocean meets the shore.",
        "Coastal beach with light sand and {n} people visible.",
    ],
    "forest": [
        "A dense forest covering a large area with {n} distinct tree clusters.",
        "Thick woodland with various shades of green vegetation.",
        "A forested region with a mix of deciduous and coniferous trees.",
    ],
    "residential": [
        "A residential neighborhood with {n} houses arranged in blocks.",
        "Suburban area with single-family homes and tree-lined streets.",
        "A housing development with {n} buildings and connecting roads.",
    ],
    "highway": [
        "A major highway with {n} lanes running through the landscape.",
        "An expressway interchange connecting multiple routes.",
        "A long stretch of highway with {n} vehicles visible.",
    ],
    "river": [
        "A river meandering through the terrain with {n} visible bends.",
        "A wide river flowing from north to south through the region.",
        "Riverside area with vegetation along {n} kilometers of banks.",
    ],
    "farmland": [
        "Agricultural farmland divided into {n} rectangular plots.",
        "Cropland with {n} distinct fields showing different growth stages.",
        "Farming area with {n} irrigation circles visible from above.",
    ],
    "parking_lot": [
        "A parking lot with approximately {n} vehicles in organized rows.",
        "A large paved parking area serving a commercial complex.",
        "Parking facility with {n} cars and clear lane markings.",
    ],
    "mountain": [
        "A mountainous region with {n} peaks and deep valleys.",
        "Rugged mountain terrain with snow on {n} higher summits.",
        "Mountain range characterized by steep slopes and {n} ridges.",
    ],
    "harbor": [
        "A busy harbor with {n} ships docked at the piers.",
        "A port area with cargo containers and {n} loading cranes.",
        "Harbor with {n} boats and warehouses along the waterfront.",
    ],
    "desert": [
        "A desert landscape with {n} distinct dune formations.",
        "Arid desert terrain with sparse vegetation and {n} sand patterns.",
        "A dry desert region with {n} visible rock outcrops.",
    ],
    "grassland": [
        "Open grassland with {n} scattered trees and rolling terrain.",
        "A prairie landscape covered in tall grasses.",
        "Grassland area with {n} grazing animals visible.",
    ],
    "industrial": [
        "An industrial zone with {n} large factory buildings.",
        "Industrial area with warehouses, smokestacks, and {n} storage tanks.",
        "A manufacturing district with {n} processing facilities.",
    ],
    "lake": [
        "A lake with clear water surrounded by {n} types of vegetation.",
        "An inland lake measuring approximately {n} square kilometers.",
        "A lake with {n} small boats and a surrounding recreation area.",
    ],
    "bridge": [
        "A bridge spanning {n} meters across the water below.",
        "A major bridge connecting two sides with {n} approach ramps.",
        "An arched bridge structure with {n} visible support pillars.",
    ],
}

# 默认兜底描述
DEFAULT_CAPTIONS = [
    "A satellite image showing {n} distinct surface features.",
    "An aerial photograph of this area with {n} visible landmarks.",
    "Remote sensing imagery depicting {n} land cover types.",
    "A high-resolution image showing the spatial layout of the region.",
    "An overhead view of this location with {n} notable features.",
]


def generate_random_image(width: int = 224, height: int = 224) -> np.ndarray:
    """生成一张随机颜色的假遥感图像。

    用简单的几何图案（渐变背景 + 色块）模拟遥感图像的视觉特征，
    而非纯随机噪声，这样更贴近真实图像分布。
    """
    img = np.random.randint(30, 200, (height, width, 3), dtype=np.uint8)

    # 模拟地物纹理：添加一些随机色块
    for _ in range(random.randint(3, 8)):
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)
        x2 = min(x1 + random.randint(20, 80), width)
        y2 = min(y1 + random.randint(20, 80), height)
        color = np.random.randint(20, 220, 3, dtype=np.uint8)
        img[y1:y2, x1:x2] = color

    return img  # HWC uint8


def generate_captions(category: str, num_captions: int = 5) -> list:
    """为指定类别生成描述文本。"""
    templates = CATEGORY_CAPTIONS.get(
        category, DEFAULT_CAPTIONS + CATEGORY_CAPTIONS.get("farmland", [])
    )
    captions = []
    for _ in range(num_captions):
        template = random.choice(templates)
        n = random.randint(2, 8)
        caption = template.format(n=n)
        captions.append(caption)
    return captions


def main():
    parser = argparse.ArgumentParser(description="生成遥感描述假数据")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="生成的样本数")
    parser.add_argument("--image_size", type=int, default=224,
                        help="图像尺寸（正方形）")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="JSONL 输出目录")
    parser.add_argument("--image_dir", type=str, default="data/raw/dummy/images",
                        help="假图像存储目录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 创建目录 ─────────────────────────
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "dummy.jsonl"

    # ── 生成样本 ─────────────────────────
    print(f"正在生成 {args.num_samples} 条假样本...")
    samples = []
    categories_list = list(CATEGORY_CAPTIONS.keys())

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            # 随机选择类别
            category = random.choice(categories_list)

            # 生成图像
            img = generate_random_image(args.image_size, args.image_size)
            image_filename = f"dummy_{i:05d}.jpg"
            image_path = image_dir / image_filename
            Image.fromarray(img).save(image_path, quality=85)

            # 生成描述
            captions = generate_captions(category)
            assert len(captions) >= 3, f"描述数量不足：{len(captions)}"

            sample = {
                "image": str(image_path.absolute()),
                "captions": captions,
                "category": category,
                "metadata": {
                    "resolution": f"{args.image_size}x{args.image_size}",
                    "source": "dummy",
                },
            }

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            samples.append(sample)

            if (i + 1) % 25 == 0:
                print(f"  已生成 {i + 1}/{args.num_samples} 条...")

    # ── 统计 ─────────────────────────────
    category_counts = {}
    for s in samples:
        category_counts[s["category"]] = category_counts.get(s["category"], 0) + 1

    print(f"\n✅ 完成！")
    print(f"   样本数: {len(samples)}")
    print(f"   类别数: {len(category_counts)}")
    print(f"   输出文件: {output_file.absolute()}")
    print(f"   图像目录: {image_dir.absolute()}")
    print(f"\n类别分布: {json.dumps(category_counts, indent=2)}")


if __name__ == "__main__":
    main()
