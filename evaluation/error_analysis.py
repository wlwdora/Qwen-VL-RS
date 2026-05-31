"""遥感图像描述的细粒度错误分析。

沿以下维度对预测错误进行分类：
  1. 地物类别：地物类型的误分类
  2. 空间关系：方向/空间描述错误
  3. 物体数量：物体计数偏差
  4. 幻觉：描述中出现了图中不存在的物体
  5. 光谱描述：颜色/光谱特性的错误描述

输出结构化的错误分类统计，用于指导后续改进方向。
"""

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """分类并量化图像描述的生成错误。"""

    def __init__(self, land_cover_vocab: Optional[set] = None):
        self.land_cover_vocab = land_cover_vocab or set()

    def analyze(
        self,
        predictions: List[str],
        references: List[List[str]],
        categories: List[str],
        image_paths: List[str],
    ) -> Dict[str, any]:
        """运行完整的错误分析。

        参数：
            predictions: 模型生成的描述列表。
            references: ground-truth 参考描述（每张图 5 句）。
            categories: 每张图对应的地物类别标签。
            image_paths: 原始图像路径，用于人工抽检。

        返回：
            包含错误分类统计和逐样本错误标记的字典。
        """
        # TODO: 实现错误分析
        # 1. 按类别统计错误分布
        # 2. 提取并比较空间关系描述
        # 3. 幻觉检测（CHAIR）
        # 4. 找到"最差 case"供人工检查
        raise NotImplementedError("【待实现】错误分析")

    def find_worst_cases(
        self, predictions: List[str], references: List[List[str]], n: int = 20
    ) -> List[Dict]:
        """返回得分最差的 n 个样本，供人工检查。"""
        # TODO: 实现最差 case 查找
        raise NotImplementedError("【待实现】最差 case 查找")
