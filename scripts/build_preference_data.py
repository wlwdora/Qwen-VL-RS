"""构建 DPO 偏好数据集。

从 RSICD 数据中构建 (chosen, rejected) 偏好对：
  - chosen (y_w): 最具区分性的参考描述（高 TF-IDF，多独特词汇）
  - rejected (y_l): 最模板化的参考描述（高频词占比高）

这直接针对 CIDEr-D 低的问题——训练模型远离"many buildings and green trees"模板。

输出格式（每行 JSON）：
{
  "image": "path/to/image.jpg",
  "prompt": "Describe this remote sensing image in detail.",
  "chosen": "detailed distinctive caption...",
  "rejected": "generic template caption...",
  "category": "airport",
  "chosen_score": 0.85,   // 区分性评分
  "rejected_score": 0.12,
}
"""

import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple


def build_tfidf_vocab(samples: List[Dict]) -> Dict[str, float]:
    """在全部样本上计算 IDF 值。"""
    n_docs = len(samples)
    doc_freq = Counter()

    for s in samples:
        words = set()
        for cap in s["captions"]:
            words.update(cap.lower().split())
        doc_freq.update(words)

    idf = {word: math.log(n_docs / (1 + freq)) for word, freq in doc_freq.items()}
    return idf


def caption_distinctiveness(caption: str, idf: Dict[str, float]) -> float:
    """计算一条 caption 的'区分性'得分。

    得分 = caption 中所有词的 IDF 平均值。
    含有稀有词（如 "crescent", "awnings"）的 caption 得分高，
    全是高频词（"many", "buildings", "trees"）的得分低。
    """
    words = caption.lower().split()
    if not words:
        return 0.0
    idf_sum = sum(idf.get(w, 0.0) for w in words)
    return idf_sum / len(words)


def caption_length_penalty(caption: str) -> float:
    """长度因子：鼓励稍长的描述，但不过度。"""
    n_words = len(caption.split())
    # 最佳长度 ~15-25 词，过短 (<8) 或过长 (>40) 扣分
    if n_words < 8:
        return n_words / 8.0
    elif n_words > 40:
        return 40.0 / n_words
    else:
        return 1.0


def score_captions(
    captions: List[str], idf: Dict[str, float]
) -> List[Tuple[str, float]]:
    """对一组 caption 按质量评分排序。"""
    scored = []
    for cap in captions:
        distinct = caption_distinctiveness(cap, idf)
        length_bonus = caption_length_penalty(cap)
        score = distinct * length_bonus
        scored.append((cap, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def build_preference_pairs(
    data_path: str,
    output_path: str,
    min_score_gap: float = 0.15,
    prompt: str = "Describe this remote sensing image in detail.",
):
    """从 RSICD JSONL 构建 DPO 偏好对。

    参数：
        data_path: 预处理后的 RSICD JSONL 路径。
        output_path: 输出的偏好对 JSONL 路径。
        min_score_gap: chosen 和 rejected 的最小分数差（避免无明显差异的对）。
        prompt: 图像描述 prompt。
    """
    # 加载数据
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"加载 {len(samples)} 条样本")

    # 计算全局 IDF
    print("计算 IDF 词汇表...")
    idf = build_tfidf_vocab(samples)

    # 构建偏好对
    pairs = []
    skipped_no_gap = 0
    skipped_single_cap = 0

    for s in samples:
        caps = s["captions"]

        # 去重
        unique_caps = list(dict.fromkeys(caps))  # 保持顺序的去重
        if len(unique_caps) < 2:
            skipped_single_cap += 1
            continue

        # 评分排序
        scored = score_captions(unique_caps, idf)

        chosen_cap, chosen_score = scored[0]
        rejected_cap, rejected_score = scored[-1]

        # 确保足够区分度
        if abs(chosen_score - rejected_score) < min_score_gap:
            skipped_no_gap += 1
            continue

        pairs.append({
            "image": s["image"],
            "prompt": prompt,
            "chosen": chosen_cap,
            "rejected": rejected_cap,
            "category": s["category"],
            "chosen_score": round(chosen_score, 4),
            "rejected_score": round(rejected_score, 4),
        })

    print(f"生成 {len(pairs)} 个偏好对")
    print(f"  跳过（单条 caption）: {skipped_single_cap}")
    print(f"  跳过（区分度不足）: {skipped_no_gap}")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"偏好数据已保存至: {output_path}")

    # ── 示例 ──
    print("\n=== 示例偏好对 ===")
    for pair in pairs[:3]:
        print(f"\n  Category: {pair['category']}")
        print(f"  Chosen   (score={pair['chosen_score']:.2f}): {pair['chosen'][:120]}")
        print(f"  Rejected (score={pair['rejected_score']:.2f}): {pair['rejected'][:120]}")

    # ── 统计 ──
    chosen_lens = [len(p["chosen"].split()) for p in pairs]
    rejected_lens = [len(p["rejected"].split()) for p in pairs]
    print(f"\n=== 统计 ===")
    print(f"  Chosen 平均长度:   {sum(chosen_lens)/len(chosen_lens):.1f} 词")
    print(f"  Rejected 平均长度: {sum(rejected_lens)/len(rejected_lens):.1f} 词")
    print(f"  训练/验证划分: 90%/10%")

    return pairs


if __name__ == "__main__":
    build_preference_pairs(
        data_path="D:/work/Qwen-VL-RS/data/processed/rsicd.jsonl",
        output_path="D:/work/Qwen-VL-RS/data/processed/rsicd_dpo_pairs.jsonl",
        min_score_gap=0.15,
    )
