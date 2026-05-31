"""遥感图像描述的细粒度错误分析。

沿以下维度对预测错误进行分类：
  1. 地物类别：地物类型的误分类
  2. 空间关系：方向/空间描述错误
  3. 物体数量：物体计数偏差
  4. 幻觉：描述中出现了图中不存在的物体
  5. 光谱描述：颜色/光谱特性的错误描述

输出结构化的错误分类统计，用于指导后续改进方向。

用法：
    >>> analyzer = ErrorAnalyzer()
    >>> report = analyzer.analyze(predictions, references, categories, image_paths)
    >>> analyzer.print_report(report)

CLI：
    python -m evaluation.error_analysis --predictions results.json --output report.json
"""

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from training.metrics import CaptioningMetrics

logger = logging.getLogger(__name__)

# ── 遥感图像常见物体词汇 ──────────────────────────
REMOTE_SENSING_OBJECTS: Set[str] = {
    # 人造结构
    "building", "buildings", "house", "houses", "road", "roads", "highway",
    "bridge", "bridges", "airport", "runway", "runways", "port", "harbor",
    "stadium", "parking", "parking lot", "tennis court", "tennis courts",
    "swimming pool", "factory", "factories", "warehouse", "warehouses",
    "storage tank", "storage tanks", "chimney", "chimneys",
    "飞机", "飞机场", "跑道", "建筑", "建筑物", "房子", "房屋", "道路", "公路",
    "桥", "桥梁", "港口", "码头", "停车场", "网球场", "游泳池",
    # 自然要素
    "tree", "trees", "forest", "forests", "grass", "meadow", "meadows",
    "river", "rivers", "lake", "lakes", "pond", "ponds", "sea", "ocean",
    "beach", "beaches", "mountain", "mountains", "hill", "hills",
    "island", "islands", "wetland", "wetlands", "desert",
    "树", "树木", "森林", "草地", "草", "河流", "河", "湖", "湖泊", "池塘",
    "海", "海滩", "山", "山脉", "丘陵", "岛屿", "湿地", "沙漠",
    # 农田/土地
    "farmland", "farm", "farms", "crop", "crops", "field", "fields",
    "orchard", "vineyard", "terrace", "terraces",
    "农田", "耕地", "农作物", "果园", "梯田",
    # 车辆
    "car", "cars", "truck", "trucks", "bus", "buses", "train", "trains",
    "ship", "ships", "boat", "boats", "plane", "planes", "airplane",
    "汽车", "卡车", "公共汽车", "火车", "船", "飞机",
}

# 空间关系词
SPATIAL_TERMS: Set[str] = {
    "left", "right", "top", "bottom", "above", "below", "beside",
    "next to", "near", "far", "between", "among", "around", "surround",
    "along", "across", "through", "north", "south", "east", "west",
    "northern", "southern", "eastern", "western", "middle", "center", "central",
    "edge", "corner", "side", "front", "back", "upper", "lower",
    "parallel", "perpendicular", "adjacent", "opposite",
    "左边", "右边", "上边", "下边", "旁边", "附近", "中间", "中心",
    "周围", "沿着", "穿过", "北", "南", "东", "西", "边缘", "角落",
}

# 数量词
QUANTITY_TERMS: Set[str] = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "many", "several", "few", "some", "a lot of", "numerous", "multiple",
    "一", "两", "三", "四", "五", "六", "七", "八", "九", "十",
    "许多", "一些", "几个", "多个", "很多", "大量",
}

# 颜色词
COLOR_TERMS: Set[str] = {
    "white", "black", "red", "green", "blue", "yellow", "brown", "gray", "grey",
    "dark", "light", "bright", "pale", "deep",
    "白色", "黑色", "红色", "绿色", "蓝色", "黄色", "棕色", "灰色",
    "深色", "浅色", "亮色",
}


@dataclass
class ErrorSample:
    """单个样本的错误分析结果。"""
    index: int
    image_path: str
    category: str
    prediction: str
    references: List[str]
    # 错误标记
    has_hallucination: bool = False
    hallucinated_words: List[str] = field(default_factory=list)
    spatial_errors: List[str] = field(default_factory=list)
    quantity_errors: List[str] = field(default_factory=list)
    color_errors: List[str] = field(default_factory=list)
    # 指标
    bleu_4: float = 0.0
    cider_d: float = 0.0
    # 综合错误严重度（0=完美, 越大越差）
    error_score: float = 0.0


class ErrorAnalyzer:
    """分类并量化图像描述的生成错误。

    使用方式：
        >>> analyzer = ErrorAnalyzer()
        >>> report = analyzer.analyze(predictions, references, categories, paths)
        >>> analyzer.print_report(report)
        >>> analyzer.export_json(report, "error_report.json")
    """

    def __init__(
        self,
        land_cover_vocab: Optional[Set[str]] = None,
        object_vocab: Optional[Set[str]] = None,
        spatial_terms: Optional[Set[str]] = None,
        color_terms: Optional[Set[str]] = None,
        quantity_terms: Optional[Set[str]] = None,
    ):
        """
        参数：
            land_cover_vocab: 自定义地物类别词汇表。为 None 时使用内置表。
            object_vocab: 自定义物体词汇表。
            spatial_terms: 自定义空间关系词汇。
            color_terms: 自定义颜色词汇。
            quantity_terms: 自定义数量词词汇。
        """
        self.land_cover_vocab = land_cover_vocab or set()
        self.object_vocab = object_vocab or REMOTE_SENSING_OBJECTS
        self.spatial_terms = spatial_terms or SPATIAL_TERMS
        self.color_terms = color_terms or COLOR_TERMS
        self.quantity_terms = quantity_terms or QUANTITY_TERMS

    # ════════════════════════════════════════════════════════════
    # 主入口
    # ════════════════════════════════════════════════════════════

    def analyze(
        self,
        predictions: List[str],
        references: List[List[str]],
        categories: List[str],
        image_paths: Optional[List[str]] = None,
    ) -> Dict:
        """运行完整的错误分析管线。

        返回结构：
        {
            "summary": {
                "total": int,
                "avg_bleu_4": float,
                "hallucination_rate": float,      # 含幻觉的样本占比
                "spatial_error_rate": float,       # 含空间错误的样本占比
                "quantity_error_rate": float,
                "color_error_rate": float,
            },
            "per_category": { category: { ... } },
            "error_types": { "hallucination": int, "spatial": int, ... },
            "worst_cases": [ ErrorSample, ... ],
            "samples": [ ErrorSample, ... ],
        }
        """
        if image_paths is None:
            image_paths = [f"sample_{i}" for i in range(len(predictions))]

        n = len(predictions)
        logger.info(f"分析 {n} 个样本...")

        # ── Step 1: 计算逐样本 BLEU-4 ────────
        sample_bleu = self._compute_per_sample_bleu(predictions, references)

        # ── Step 2: 逐样本错误分类 ──────────
        error_samples: List[ErrorSample] = []
        for i in range(n):
            pred = predictions[i]
            refs = references[i]
            cat = categories[i] if i < len(categories) else "unknown"
            path = image_paths[i] if i < len(image_paths) else ""

            es = ErrorSample(
                index=i,
                image_path=path,
                category=cat,
                prediction=pred,
                references=refs,
                bleu_4=sample_bleu[i],
            )

            # 幻觉检测
            es.has_hallucination, es.hallucinated_words = self._detect_hallucination(
                pred, refs
            )

            # 空间关系错误
            es.spatial_errors = self._compare_spatial(pred, refs)

            # 数量错误
            es.quantity_errors = self._compare_quantities(pred, refs)

            # 颜色错误
            es.color_errors = self._compare_colors(pred, refs)

            # 综合错误得分
            es.error_score = self._compute_error_score(es)

            error_samples.append(es)

        # ── Step 3: 汇总统计 ──────────────────
        summary = self._build_summary(error_samples)

        # ── Step 4: 按类别统计 ────────────────
        per_category = self._build_per_category(error_samples)

        # ── Step 5: 错误类型分布 ──────────────
        error_type_counts = self._count_error_types(error_samples)

        # ── Step 6: 最差 cases ────────────────
        worst_cases = self.find_worst_cases(predictions, references, image_paths,
                                            categories, n=20)

        return {
            "summary": summary,
            "per_category": per_category,
            "error_types": error_type_counts,
            "worst_cases": worst_cases,
            "samples": [self._error_sample_to_dict(es) for es in error_samples],
        }

    # ════════════════════════════════════════════════════════════
    # 错误检测子方法
    # ════════════════════════════════════════════════════════════

    def _compute_per_sample_bleu(
        self, predictions: List[str], references: List[List[str]]
    ) -> List[float]:
        """逐样本计算 BLEU-4 得分。"""
        scores = []
        try:
            import io
            import contextlib
            from pycocoevalcap.bleu.bleu import Bleu
            scorer = Bleu(4)
            for pred, refs in zip(predictions, references):
                gts = {0: refs}
                res = {0: [pred]}
                with contextlib.redirect_stdout(io.StringIO()):
                    _, s = scorer.compute_score(gts, res)
                if isinstance(s, list) and len(s) > 0 and len(s[0]) > 0:
                    scores.append(float(s[0][0]))
                else:
                    scores.append(0.0)
        except ImportError:
            scores = [0.0] * len(predictions)
        return scores

    def _detect_hallucination(
        self, prediction: str, references: List[str]
    ) -> Tuple[bool, List[str]]:
        """检测预测句中的幻觉物体（预测提到但参考句中都未提及的物体）。

        返回：
            (has_hallucination, hallucinated_words)
        """
        pred_nouns = CaptioningMetrics._extract_nouns(prediction)

        # 收集所有参考句中出现的物体
        ref_nouns_all: Set[str] = set()
        for ref in references:
            ref_nouns_all.update(CaptioningMetrics._extract_nouns(ref))

        # 仅在参考句中出现过的才算有效物体，否则可能是幻觉
        hallucinated = []
        for word in pred_nouns:
            if word not in ref_nouns_all and word in self.object_vocab:
                hallucinated.append(word)

        return len(hallucinated) > 0, hallucinated

    def _compare_spatial(
        self, prediction: str, references: List[str]
    ) -> List[str]:
        """比较空间关系描述的差异。

        简化判断：预测中出现的空间词在参考句中出现过至少一次则算正确。
        返回：预测中存在但参考句均不存在的空间词列表。
        """
        pred_spatial = {w for w in self.spatial_terms if w in prediction.lower()}
        ref_spatial: Set[str] = set()
        for ref in references:
            ref_spatial.update(w for w in self.spatial_terms if w in ref.lower())

        errors = list(pred_spatial - ref_spatial)
        return errors

    def _compare_quantities(
        self, prediction: str, references: List[str]
    ) -> List[str]:
        """比较数量描述的差异。"""
        pred_quant = {w for w in self.quantity_terms if w in prediction.lower()}
        ref_quant: Set[str] = set()
        for ref in references:
            ref_quant.update(w for w in self.quantity_terms if w in ref.lower())

        errors = list(pred_quant - ref_quant)
        return errors

    def _compare_colors(
        self, prediction: str, references: List[str]
    ) -> List[str]:
        """比较颜色描述的差异。"""
        pred_colors = {w for w in self.color_terms if w in prediction.lower()}
        ref_colors: Set[str] = set()
        for ref in references:
            ref_colors.update(w for w in self.color_terms if w in ref.lower())

        errors = list(pred_colors - ref_colors)
        return errors

    def _compute_error_score(self, es: ErrorSample) -> float:
        """计算综合错误严重度得分。

        得分构成：
          - BLEU-4 贡献：1 - BLEU-4（BLEU 越低，得分越高）
          - 幻觉：每个幻觉词 +0.3
          - 空间错误：+0.2
          - 数量错误：+0.15
          - 颜色错误：+0.1
        """
        score = (1.0 - es.bleu_4) if es.bleu_4 > 0 else 1.0
        score += len(es.hallucinated_words) * 0.3
        score += len(es.spatial_errors) * 0.2
        score += len(es.quantity_errors) * 0.15
        score += len(es.color_errors) * 0.1
        return round(score, 3)

    # ════════════════════════════════════════════════════════════
    # 汇总与统计
    # ════════════════════════════════════════════════════════════

    def _build_summary(self, error_samples: List[ErrorSample]) -> Dict:
        """构建全局汇总统计。"""
        n = len(error_samples)
        if n == 0:
            return {}

        avg_bleu = np.mean([es.bleu_4 for es in error_samples])
        hallucination_rate = sum(1 for es in error_samples if es.has_hallucination) / n
        spatial_err_rate = sum(1 for es in error_samples if es.spatial_errors) / n
        quantity_err_rate = sum(1 for es in error_samples if es.quantity_errors) / n
        color_err_rate = sum(1 for es in error_samples if es.color_errors) / n

        avg_hallucinated = np.mean([len(es.hallucinated_words) for es in error_samples])
        avg_error_score = np.mean([es.error_score for es in error_samples])

        return {
            "total_samples": n,
            "avg_bleu_4": round(float(avg_bleu), 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "spatial_error_rate": round(spatial_err_rate, 4),
            "quantity_error_rate": round(quantity_err_rate, 4),
            "color_error_rate": round(color_err_rate, 4),
            "avg_hallucinated_words_per_sample": round(float(avg_hallucinated), 2),
            "avg_error_score": round(float(avg_error_score), 3),
        }

    def _build_per_category(
        self, error_samples: List[ErrorSample]
    ) -> Dict[str, Dict]:
        """按地物类别汇总错误统计。"""
        cat_samples: Dict[str, List[ErrorSample]] = defaultdict(list)
        for es in error_samples:
            cat_samples[es.category].append(es)

        per_category = {}
        for cat, samples in sorted(cat_samples.items()):
            n = len(samples)
            per_category[cat] = {
                "count": n,
                "avg_bleu_4": round(float(np.mean([s.bleu_4 for s in samples])), 4),
                "hallucination_rate": round(
                    sum(1 for s in samples if s.has_hallucination) / n, 4
                ),
                "spatial_error_rate": round(
                    sum(1 for s in samples if s.spatial_errors) / n, 4
                ),
                "avg_error_score": round(
                    float(np.mean([s.error_score for s in samples])), 3
                ),
            }

        return per_category

    def _count_error_types(
        self, error_samples: List[ErrorSample]
    ) -> Dict[str, int]:
        """统计各错误类型的出现次数。"""
        counts = {
            "hallucination": 0,
            "spatial": 0,
            "quantity": 0,
            "color": 0,
            "total_errors": 0,
        }
        for es in error_samples:
            if es.has_hallucination:
                counts["hallucination"] += 1
            if es.spatial_errors:
                counts["spatial"] += 1
            if es.quantity_errors:
                counts["quantity"] += 1
            if es.color_errors:
                counts["color"] += 1
        counts["total_errors"] = (
            counts["hallucination"]
            + counts["spatial"]
            + counts["quantity"]
            + counts["color"]
        )
        return counts

    def _error_sample_to_dict(self, es: ErrorSample) -> Dict:
        """将 ErrorSample 转为可序列化字典。"""
        return {
            "index": es.index,
            "image_path": es.image_path,
            "category": es.category,
            "prediction": es.prediction,
            "references": es.references,
            "has_hallucination": es.has_hallucination,
            "hallucinated_words": es.hallucinated_words,
            "spatial_errors": es.spatial_errors,
            "quantity_errors": es.quantity_errors,
            "color_errors": es.color_errors,
            "bleu_4": round(es.bleu_4, 4),
            "cider_d": es.cider_d,
            "error_score": es.error_score,
        }

    # ════════════════════════════════════════════════════════════
    # 最差 Case 查找
    # ════════════════════════════════════════════════════════════

    def find_worst_cases(
        self,
        predictions: List[str],
        references: List[List[str]],
        image_paths: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        n: int = 20,
    ) -> List[Dict]:
        """返回得分最差的 n 个样本，供人工检查。

        排序依据：综合 error_score（越高越差），同分时按 BLEU-4（越低越差）。
        """
        # 先跑一次完整分析获取逐样本得分
        if image_paths is None:
            image_paths = [f"sample_{i}" for i in range(len(predictions))]

        error_samples = []
        sample_bleu = self._compute_per_sample_bleu(predictions, references)

        for i in range(len(predictions)):
            pred = predictions[i]
            refs = references[i]
            cat = categories[i] if categories and i < len(categories) else "unknown"
            path = image_paths[i] if i < len(image_paths) else ""

            # 幻觉检测
            has_hallu, hallu_words = self._detect_hallucination(pred, refs)
            spatial_errs = self._compare_spatial(pred, refs)
            quant_errs = self._compare_quantities(pred, refs)
            color_errs = self._compare_colors(pred, refs)

            es = ErrorSample(
                index=i,
                image_path=path,
                category=cat,
                prediction=pred,
                references=refs,
                bleu_4=sample_bleu[i],
                has_hallucination=has_hallu,
                hallucinated_words=hallu_words,
                spatial_errors=spatial_errs,
                quantity_errors=quant_errs,
                color_errors=color_errs,
                error_score=0.0,
            )
            es.error_score = self._compute_error_score(es)
            error_samples.append(es)

        # 按 error_score 降序（越差越前）
        error_samples.sort(key=lambda x: (x.error_score, -x.bleu_4), reverse=True)
        worst = error_samples[:n]

        return [self._error_sample_to_dict(es) for es in worst]

    def find_best_cases(
        self,
        predictions: List[str],
        references: List[List[str]],
        categories: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        n: int = 20,
    ) -> List[Dict]:
        """返回得分最好的 n 个样本。"""
        if image_paths is None:
            image_paths = [f"sample_{i}" for i in range(len(predictions))]

        sample_bleu = self._compute_per_sample_bleu(predictions, references)

        paired = []
        for i in range(len(predictions)):
            paired.append({
                "index": i,
                "image_path": image_paths[i] if i < len(image_paths) else "",
                "category": categories[i] if categories and i < len(categories) else "unknown",
                "prediction": predictions[i],
                "references": references[i],
                "bleu_4": round(sample_bleu[i], 4),
            })

        paired.sort(key=lambda x: x["bleu_4"], reverse=True)
        return paired[:n]

    # ════════════════════════════════════════════════════════════
    # 报告输出
    # ════════════════════════════════════════════════════════════

    def print_report(self, analysis: Dict, show_worst: int = 10):
        """打印人类可读的错误分析报告。"""
        summary = analysis.get("summary", {})
        per_cat = analysis.get("per_category", {})
        error_types = analysis.get("error_types", {})
        worst = analysis.get("worst_cases", [])

        print("\n" + "=" * 70)
        print("  遥感图像描述 — 错误分析报告")
        print("=" * 70)

        # ── 全局汇总 ──────────────────────
        print(f"\n  【全局统计】({summary.get('total_samples', 0)} 样本)")
        print(f"    平均 BLEU-4:          {summary.get('avg_bleu_4', 0):.4f}")
        print(f"    幻觉率:               {summary.get('hallucination_rate', 0)*100:.1f}%")
        print(f"    空间关系错误率:       {summary.get('spatial_error_rate', 0)*100:.1f}%")
        print(f"    数量描述错误率:       {summary.get('quantity_error_rate', 0)*100:.1f}%")
        print(f"    颜色描述错误率:       {summary.get('color_error_rate', 0)*100:.1f}%")
        print(f"    平均每样本幻觉词数:   {summary.get('avg_hallucinated_words_per_sample', 0):.1f}")
        print(f"    平均错误严重度:       {summary.get('avg_error_score', 0):.3f}")

        # ── 错误类型分布 ──────────────────
        print(f"\n  【错误类型分布】")
        for err_type, count in error_types.items():
            if err_type == "total_errors":
                continue
            pct = count / summary.get('total_samples', 1) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"    {err_type:15s} {count:4d} ({pct:5.1f}%) {bar}")

        # ── 按类别 ──────────────────────────
        if per_cat:
            print(f"\n  【按类别统计】（前 10 / 后 5）")
            # 按 avg_bleu_4 排序
            cat_ranked = sorted(per_cat.items(), key=lambda x: x[1]["avg_bleu_4"], reverse=True)

            print(f"\n    最佳类别（BLEU-4 最高）:")
            print(f"    {'类别':20s} {'样本':>5s} {'BLEU-4':>8s} {'幻觉率':>7s} {'错误分':>7s}")
            print(f"    {'-'*50}")
            for cat, stats in cat_ranked[:10]:
                print(f"    {cat:20s} {stats['count']:5d} {stats['avg_bleu_4']:8.4f} "
                      f"{stats['hallucination_rate']*100:6.1f}% {stats['avg_error_score']:7.3f}")

            print(f"\n    最差类别（BLEU-4 最低）:")
            print(f"    {'类别':20s} {'样本':>5s} {'BLEU-4':>8s} {'幻觉率':>7s} {'错误分':>7s}")
            print(f"    {'-'*50}")
            for cat, stats in cat_ranked[-5:]:
                print(f"    {cat:20s} {stats['count']:5d} {stats['avg_bleu_4']:8.4f} "
                      f"{stats['hallucination_rate']*100:6.1f}% {stats['avg_error_score']:7.3f}")

        # ── 最差 cases ─────────────────────
        if worst:
            print(f"\n  【最差 {min(show_worst, len(worst))} 个样本】")
            for i, case in enumerate(worst[:show_worst]):
                print(f"\n    #{i+1} [{case['category']}] — error_score={case['error_score']:.3f}")
                print(f"      Ref : {case['references'][0][:100]}")
                print(f"      Pred: {case['prediction'][:100]}")
                if case.get("hallucinated_words"):
                    print(f"      幻觉词: {', '.join(case['hallucinated_words'])}")
                if case.get("spatial_errors"):
                    print(f"      空间错误: {', '.join(case['spatial_errors'])}")

        print("\n" + "=" * 70)

    def export_json(self, analysis: Dict, output_path: str):
        """将分析报告导出为 JSON 文件。"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # 确保所有值可序列化
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        logger.info(f"错误分析报告已保存至: {output_path}")

    def export_markdown(self, analysis: Dict, output_path: str):
        """将分析报告导出为 Markdown 文件。"""
        summary = analysis.get("summary", {})
        per_cat = analysis.get("per_category", {})
        error_types = analysis.get("error_types", {})
        worst = analysis.get("worst_cases", [])

        lines = []
        lines.append("# 遥感图像描述 — 错误分析报告\n")

        lines.append("## 全局统计\n")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|:---|:---|")
        lines.append(f"| 样本数 | {summary.get('total_samples', 0)} |")
        lines.append(f"| 平均 BLEU-4 | {summary.get('avg_bleu_4', 0):.4f} |")
        lines.append(f"| 幻觉率 | {summary.get('hallucination_rate', 0)*100:.1f}% |")
        lines.append(f"| 空间关系错误率 | {summary.get('spatial_error_rate', 0)*100:.1f}% |")
        lines.append(f"| 平均错误严重度 | {summary.get('avg_error_score', 0):.3f} |")

        lines.append("\n## 错误类型分布\n")
        lines.append(f"| 错误类型 | 样本数 | 占比 |")
        lines.append(f"|:---|:---|:---|")
        for err_type, count in error_types.items():
            if err_type == "total_errors":
                continue
            pct = count / summary.get('total_samples', 1) * 100
            lines.append(f"| {err_type} | {count} | {pct:.1f}% |")

        if per_cat:
            lines.append("\n## 按类别统计\n")
            cat_ranked = sorted(per_cat.items(), key=lambda x: x[1]["avg_bleu_4"], reverse=True)
            lines.append(f"| 类别 | 样本数 | BLEU-4 | 幻觉率 | 错误分 |")
            lines.append(f"|:---|:---|:---|:---|:---|")
            for cat, stats in cat_ranked:
                lines.append(
                    f"| {cat} | {stats['count']} | {stats['avg_bleu_4']:.4f} | "
                    f"{stats['hallucination_rate']*100:.1f}% | {stats['avg_error_score']:.3f} |"
                )

        if worst:
            lines.append(f"\n## 最差 {len(worst)} 个样本\n")
            for i, case in enumerate(worst):
                lines.append(f"### #{i+1} [{case['category']}] — error_score={case['error_score']:.3f}\n")
                lines.append(f"- **参考**: {case['references'][0][:200]}")
                lines.append(f"- **预测**: {case['prediction'][:200]}")
                if case.get("hallucinated_words"):
                    lines.append(f"- **幻觉词**: {', '.join(case['hallucinated_words'])}")
                lines.append("")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Markdown 报告已保存至: {output_path}")


# ════════════════════════════════════════════════════════════════
# CLI 入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="遥感图像描述错误分析")
    parser.add_argument("--predictions", type=str, required=True,
                        help="预测结果 JSON 文件路径")
    parser.add_argument("--output", type=str, default="experiments/error_report.json",
                        help="输出报告路径")
    parser.add_argument("--format", type=str, default="json",
                        choices=["json", "markdown"],
                        help="输出格式")

    args = parser.parse_args()

    with open(args.predictions, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = [s["prediction"] for s in data["samples"]]
    references = [s["references"] for s in data["samples"]]
    categories = [s.get("category", "unknown") for s in data["samples"]]
    image_paths = [s.get("image_path", "") for s in data["samples"]]

    analyzer = ErrorAnalyzer()
    report = analyzer.analyze(predictions, references, categories, image_paths)
    analyzer.print_report(report)

    if args.format == "json":
        analyzer.export_json(report, args.output)
    else:
        analyzer.export_markdown(report, args.output.replace(".json", ".md"))

    print(f"\n报告已保存至: {args.output}")
