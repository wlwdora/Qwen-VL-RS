"""遥感图像描述评估指标。

标准图像描述指标（基于 pycocoevalcap）：
  - BLEU-1/2/3/4：n-gram 精确率 + 长度惩罚
  - METEOR：unigram 精确率/召回率 + 同义词匹配
  - ROUGE-L：基于最长公共子序列的召回率
  - CIDEr-D：TF-IDF 加权的共识度量（对图像描述最敏感，是首选指标）
  - SPICE：基于场景图的语义命题评估

遥感领域特定指标（自实现）：
  - CHAIR-s / CHAIR-i：图像描述幻觉评估
  - Land Cover F1：地物类别词汇的精确率/召回率/F1

用法：
    >>> metrics = CaptioningMetrics(metrics=["bleu", "cider"])
    >>> results = metrics.compute(predictions, references)
    >>> print(results)  # {"bleu_4": 26.5, "cider_d": 82.3}
"""

import json
import logging
import re
import tempfile
import os
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CaptioningMetrics:
    """图像描述评估指标的计算与聚合。

    使用 COCO 评估协议：每个样本有 1 个预测描述 vs 多个人工参考描述。
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        参数：
            metrics: 要计算的指标名称列表。
                     可选：bleu, meteor, rouge, cider, spice, chair, landcover_f1
                     默认：["bleu", "meteor", "rouge", "cider", "spice"]
        """
        self.metrics = metrics or ["bleu", "meteor", "rouge", "cider", "spice"]

    def compute(
        self, predictions: List[str], references: List[List[str]]
    ) -> Dict[str, float]:
        """计算所有配置的指标。

        参数：
            predictions: 模型生成的描述列表，长度为 N。
            references: 参考描述列表的列表，shape 为 N × K（K 条参考描述/样本）。

        返回：
            指标名 → 得分 的字典。如 {"bleu_4": 26.5, "cider_d": 82.3, ...}
        """
        results = {}

        # ── 构建 COCO 格式 ─────────────────────────
        # gts: {id: [ref1, ref2, ...]}
        # res: {id: [pred]}
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}

        # ── 逐指标计算 ────────────────────────────
        for metric_name in self.metrics:
            metric_lower = metric_name.lower()

            if metric_lower == "bleu":
                scores = self._compute_bleu(gts, res)
                results.update(scores)

            elif metric_lower == "meteor":
                score = self._compute_meteor(gts, res)
                if score is not None:
                    results["meteor"] = score

            elif metric_lower == "rouge":
                score = self._compute_rouge(gts, res)
                if score is not None:
                    results["rouge_l"] = score

            elif metric_lower == "cider":
                score = self._compute_cider(gts, res)
                if score is not None:
                    results["cider_d"] = score

            elif metric_lower == "spice":
                score = self._compute_spice(gts, res)
                if score is not None:
                    results["spice"] = score

            elif metric_lower == "chair":
                logger.warning(
                    "CHAIR 需要额外参数，请调用 CaptioningMetrics.compute_chair()"
                )

            elif metric_lower == "landcover_f1":
                logger.warning(
                    "Land Cover F1 需要额外参数，请调用 CaptioningMetrics.compute_land_cover_f1()"
                )

            else:
                logger.warning(f"未知指标：{metric_name}，已跳过")

        return results

    # ════════════════════════════════════════════════════════════
    # 标准指标（通过 pycocoevalcap）
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_bleu(gts: Dict, res: Dict) -> Dict[str, float]:
        """计算 BLEU-1 到 BLEU-4。"""
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            scorer = Bleu(4)
            score, scores = scorer.compute_score(gts, res)
            # score 是 [bleu1, bleu2, bleu3, bleu4] 的列表
            return {
                "bleu_1": float(score[0]),
                "bleu_2": float(score[1]),
                "bleu_3": float(score[2]),
                "bleu_4": float(score[3]),
            }
        except ImportError:
            logger.debug("pycocoevalcap 未安装，使用简化 BLEU")
            return CaptioningMetrics._compute_bleu_simple(gts, res)

    @staticmethod
    def _compute_meteor(gts: Dict, res: Dict) -> Optional[float]:
        """计算 METEOR 得分。

        注意：METEOR 需要 Java 运行时环境。Windows 上通常没有 Java，
        且 pycocoevalcap 的 Meteor.__del__ 在 __init__ 失败时会访问
        不存在的 self.lock 导致二次异常。此处做了防御性处理。
        """
        scorer = None
        try:
            import shutil
            if shutil.which("java") is None:
                logger.debug("METEOR 需要 Java 运行时，当前系统未安装，跳过")
                return None

            from pycocoevalcap.meteor.meteor import Meteor
            scorer = Meteor()
            score, scores = scorer.compute_score(gts, res)
            return float(score)
        except ImportError:
            logger.debug("pycocoevalcap 未安装，METEOR 暂不可用")
            return None
        except Exception as e:
            logger.warning(f"METEOR 计算失败：{e}")
            # 如果 __init__ 失败，Meteor 对象的状态不完整，
            # 手动置 None 避免 __del__ 时访问不存在的 lock
            if scorer is not None:
                scorer.lock = None
            return None

    @staticmethod
    def _compute_rouge(gts: Dict, res: Dict) -> Optional[float]:
        """计算 ROUGE-L 得分。"""
        try:
            from pycocoevalcap.rouge.rouge import Rouge
            scorer = Rouge()
            score, scores = scorer.compute_score(gts, res)
            return float(score)
        except ImportError:
            logger.debug("pycocoevalcap 未安装，使用简化 ROUGE-L")
            return CaptioningMetrics._compute_rouge_simple(gts, res)
        except Exception as e:
            logger.warning(f"ROUGE 计算失败：{e}")
            return None

    @staticmethod
    def _compute_cider(gts: Dict, res: Dict) -> Optional[float]:
        """计算 CIDEr-D 得分（图像描述评估的首选指标）。"""
        try:
            from pycocoevalcap.cider.cider import Cider
            scorer = Cider()
            score, scores = scorer.compute_score(gts, res)
            return float(score)
        except ImportError:
            logger.debug("pycocoevalcap 未安装，CIDEr 暂不可用")
            return None
        except Exception as e:
            logger.warning(f"CIDEr 计算失败：{e}")
            return None

    @staticmethod
    def _compute_spice(gts: Dict, res: Dict) -> Optional[float]:
        """计算 SPICE 得分。"""
        try:
            from pycocoevalcap.spice.spice import Spice
            scorer = Spice()
            score, scores = scorer.compute_score(gts, res)
            return float(score)
        except ImportError:
            logger.debug("pycocoevalcap 未安装，SPICE 暂不可用")
            return None
        except Exception as e:
            logger.warning(f"SPICE 计算失败：{e}")
            return None

    # ════════════════════════════════════════════════════════════
    # 简化后备实现（无 pycocoevalcap 时使用 nltk）
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_bleu_simple(gts: Dict, res: Dict) -> Dict[str, float]:
        """使用 NLTK 的简化 BLEU 实现（不需要 Java / pycocoevalcap）。"""
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        except ImportError:
            logger.warning("NLTK 未安装，BLEU 不可用")
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

        smooth = SmoothingFunction().method1
        list_of_refs = []
        hyps = []
        for i in sorted(gts.keys()):
            refs = gts[i]
            tokenized_refs = [r.split() for r in refs]
            list_of_refs.append(tokenized_refs)
            hyps.append(res[i][0].split() if res[i] else [])

        results = {}
        for n in [1, 2, 3, 4]:
            weights = tuple([1.0 / n] * n)  # 均匀加权
            try:
                score = corpus_bleu(list_of_refs, hyps, weights=weights, smoothing_function=smooth)
                results[f"bleu_{n}"] = float(score * 100)
            except Exception:
                results[f"bleu_{n}"] = 0.0

        return results

    @staticmethod
    def _compute_rouge_simple(gts: Dict, res: Dict) -> Optional[float]:
        """使用 Python 原生实现的简化 ROUGE-L。"""
        try:
            scores = []
            for i in sorted(gts.keys()):
                pred = res[i][0].split() if res[i] else []
                best = 0.0
                for ref in gts[i]:
                    ref_tokens = ref.split()
                    lcs_len = CaptioningMetrics._lcs_length(pred, ref_tokens)
                    if len(pred) == 0 or len(ref_tokens) == 0:
                        continue
                    p = lcs_len / len(pred) if len(pred) > 0 else 0
                    r = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0
                    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                    best = max(best, f)
                scores.append(best)
            return float(np.mean(scores) * 100) if scores else 0.0
        except Exception as e:
            logger.warning(f"简化 ROUGE-L 计算失败：{e}")
            return None

    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
        """最长公共子序列长度（DP 实现）。"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if x[i] == y[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[m][n]

    # ════════════════════════════════════════════════════════════
    # 遥感领域特定指标
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def compute_chair(
        predictions: List[str],
        references: List[List[str]],
        image_objects: List[Set[str]],
    ) -> Dict[str, float]:
        """计算 CHAIR（Caption Hallucination Assessment with Image References）指标。

        CHAIR-s：包含至少 1 个幻觉物体的句子占总句子的比例。
        CHAIR-i：所有被提及物体中幻觉物体的比例。

        参数：
            predictions: 模型生成的描述。
            references: 人工参考描述。
            image_objects: 每张图中实际包含的物体集合（从参考描述中提取）。

        返回：
            {"chair_s": float, "chair_i": float, "hallucinated_objects": List[str]}

        参考资料：Rohrbach et al., "Object Hallucination in Image Captioning", EMNLP 2018
        """
        n_sentences = len(predictions)
        hallucinated_sentences = 0
        total_objects_mentioned = 0
        total_hallucinated = 0
        all_hallucinated_objs = []

        for pred, refs, gt_objects in zip(predictions, references, image_objects):
            # 从预测描述中提取物体名词（简化：提取常见名词短语）
            pred_objects = CaptioningMetrics._extract_nouns(pred)

            total_objects_mentioned += len(pred_objects)

            # 找出不在 GT 中的物体
            hallucinated = [obj for obj in pred_objects if obj not in gt_objects]
            total_hallucinated += len(hallucinated)
            all_hallucinated_objs.extend(hallucinated)

            if len(hallucinated) > 0:
                hallucinated_sentences += 1

        chair_s = (hallucinated_sentences / n_sentences * 100) if n_sentences > 0 else 0.0
        chair_i = (
            total_hallucinated / total_objects_mentioned * 100
        ) if total_objects_mentioned > 0 else 0.0

        return {
            "chair_s": round(chair_s, 2),
            "chair_i": round(chair_i, 2),
        }

    @staticmethod
    def _extract_nouns(text: str) -> Set[str]:
        """从文本中提取物体名词（简化版——基于规则而非 NLP 解析器）。

        提取模式：
          - 常见地物名词列表匹配
          - 连续名词短语（简化为 1-3 gram）
        """
        # 地物名词词典（精简版，可扩展）
        land_cover_nouns = {
            "airport", "runway", "terminal", "airplane", "hangar",
            "beach", "coast", "shore", "sand", "ocean", "sea",
            "forest", "tree", "woodland", "vegetation",
            "building", "house", "residential", "apartment", "roof",
            "road", "highway", "street", "intersection", "bridge",
            "river", "lake", "pond", "water", "stream", "canal",
            "farmland", "crop", "field", "agriculture", "orchard",
            "parking", "car", "vehicle", "truck",
            "mountain", "hill", "valley",
            "harbor", "port", "dock", "ship", "boat",
            "stadium", "track", "court",
            "desert", "grassland", "meadow", "shrub",
            "industrial", "factory", "warehouse", "storage",
        }

        words = set(re.findall(r'\b[a-z]+\b', text.lower()))
        return words & land_cover_nouns

    @staticmethod
    def compute_land_cover_f1(
        predictions: List[str],
        references: List[List[str]],
        land_cover_vocab: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """计算地物类别词汇的精确率/召回率/F1。

        评估模型是否使用了正确的地物类别术语。

        参数：
            predictions: 模型生成的描述。
            references: 人工参考描述。
            land_cover_vocab: 地物类别词汇表。为 None 时使用内置词汇表。

        返回：
            {"landcover_precision": float, "landcover_recall": float, "landcover_f1": float}
        """
        if land_cover_vocab is None:
            land_cover_vocab = {
                "airport", "beach", "forest", "building", "road", "river",
                "lake", "farmland", "parking", "mountain", "harbor",
                "desert", "grassland", "industrial", "residential",
                "bridge", "runway", "tree", "water", "field", "house",
            }

        total_pred_matches = 0
        total_pred_terms = 0
        total_ref_terms = 0

        for pred, refs in zip(predictions, references):
            pred_terms = set(re.findall(r'\b[a-z]+\b', pred.lower())) & land_cover_vocab

            # 合并所有参考描述中的地物术语
            ref_terms = set()
            for ref in refs:
                ref_terms |= set(re.findall(r'\b[a-z]+\b', ref.lower())) & land_cover_vocab

            total_pred_terms += len(pred_terms)
            total_ref_terms += len(ref_terms)
            total_pred_matches += len(pred_terms & ref_terms)

        precision = total_pred_matches / total_pred_terms if total_pred_terms > 0 else 0.0
        recall = total_pred_matches / total_ref_terms if total_ref_terms > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "landcover_precision": round(precision * 100, 2),
            "landcover_recall": round(recall * 100, 2),
            "landcover_f1": round(f1 * 100, 2),
        }

    # ════════════════════════════════════════════════════════════
    # 辅助方法
    # ════════════════════════════════════════════════════════════

    def compute_all_detailed(
        self,
        predictions: List[str],
        references: List[List[str]],
        categories: Optional[List[str]] = None,
    ) -> Dict:
        """计算所有指标并按类别细分。

        参数：
            predictions: 模型生成的描述。
            references: 人工参考描述。
            categories: 每张图对应的地物类别标签。

        返回：
            包含全局指标和各类别指标的结构化字典。
        """
        # 全局指标
        global_results = self.compute(predictions, references)

        output = {"global": global_results}

        # 逐类别指标
        if categories is not None:
            unique_cats = set(categories)
            per_category = {}
            for cat in sorted(unique_cats):
                indices = [i for i, c in enumerate(categories) if c == cat]
                if len(indices) < 3:  # 样本太少跳过
                    continue
                cat_preds = [predictions[i] for i in indices]
                cat_refs = [references[i] for i in indices]
                per_category[cat] = self.compute(cat_preds, cat_refs)

            output["per_category"] = per_category

        return output

    @staticmethod
    def format_results(results: Dict[str, float]) -> str:
        """将指标结果格式化为可打印的表格形式。"""
        lines = []
        lines.append("-" * 42)
        lines.append(f"{'Metric':<20} {'Score':>10}")
        lines.append("-" * 42)

        # 排序：优先显示 CIDEr 和 BLEU-4
        priority = ["cider_d", "bleu_4", "meteor", "rouge_l", "spice"]
        ordered_keys = [k for k in priority if k in results]
        ordered_keys += [k for k in sorted(results) if k not in priority]

        for key in ordered_keys:
            lines.append(f"{key:<20} {results[key]:>10.2f}")

        lines.append("-" * 42)
        return "\n".join(lines)
