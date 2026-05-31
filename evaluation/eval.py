"""遥感图像描述模型评估引擎。

支持：
  - 单 checkpoint 评估（生成 + 指标计算）
  - 跨数据集 benchmark
  - 与 GPT-4V baseline 对比
  - 逐类别细分评估
  - 导出结构化 JSON 结果

用法：
    python -m evaluation.eval --checkpoint output/lora_adapter --datasets rsicd,ucm

或程序化调用：
    >>> evaluator = CaptionEvaluator(model_path="output/lora_adapter")
    >>> results = evaluator.evaluate()
    >>> evaluator.print_report(results)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

from models.qwen_vl_rs import QwenVLForRemoteSensing
from data.dataset import RemoteSensingCaptionDataset
from data.collator import MultiModalDataCollator
from training.metrics import CaptioningMetrics

logger = logging.getLogger(__name__)


class CaptionEvaluator:
    """对训练好的模型进行标准评估。

    评估流程：
      1. 加载模型（基座 + LoRA adapter）
      2. 加载测试数据集
      3. 逐批生成描述
      4. 计算所有配置的指标
      5. 输出格式化结果
    """

    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        dataset_names: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        device: str = "cuda",
        data_dir: str = "data/processed",
    ):
        """
        参数：
            model_path: 训练好的模型路径（完整模型或 LoRA adapter 路径）。
            base_model_path: 基座模型路径（当 model_path 仅为 LoRA adapter 时需要）。
                             为 None 时从 config 推断。
            dataset_names: 要评估的数据集列表。
            metrics: 指标列表。
            device: "cuda" 或 "cpu"。
            data_dir: 存放处理后数据的目录。
        """
        self.model_path = model_path
        self.base_model_path = base_model_path or self._infer_base_model()
        self.dataset_names = dataset_names or ["rsicd", "ucm", "sydney"]
        self.metric_names = metrics or ["bleu", "meteor", "rouge", "cider", "spice"]
        self.device = device
        self.data_dir = data_dir

        # ── 运行时 ─────────────────────────────────
        self.model_wrapper: Optional[QwenVLForRemoteSensing] = None
        self.metrics_calculator = CaptioningMetrics(metrics=self.metric_names)

    def _infer_base_model(self) -> str:
        """推断基座模型路径。"""
        # 优先从 config 读取
        possible = [
            "D:/Qwen/Qwen3-VL-2B-Instruct",
            "Qwen/Qwen3-VL-2B-Instruct",
        ]
        for p in possible:
            if os.path.exists(p) or "/" in p:  # HF model ID 也算有效
                return p
        return possible[0]

    def evaluate(
        self,
        batch_size: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        save_predictions: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """在所有配置的数据集上运行评估。

        返回：
            {数据集名称: {指标名称: 得分}}
            例如：{"rsicd": {"bleu_4": 26.5, "cider_d": 82.3}, ...}
        """
        self._load_model()
        results = {}

        for dataset_name in self.dataset_names:
            logger.info(f"\n{'='*50}")
            logger.info(f"正在评估：{dataset_name}")
            logger.info(f"{'='*50}")

            # ── 加载数据 ───────────────────────
            data_file = os.path.join(self.data_dir, f"{dataset_name}.jsonl")
            if not os.path.exists(data_file):
                logger.warning(f"数据文件不存在：{data_file}，跳过")
                continue

            dataset = RemoteSensingCaptionDataset(
                data_paths=data_file,
                split="test",
                max_length=512,
            )

            # ── 生成描述 ───────────────────────
            predictions, references, categories, image_paths = self._generate_all(
                dataset, batch_size, max_new_tokens, temperature
            )

            # ── 计算指标 ───────────────────────
            dataset_results = self.metrics_calculator.compute_all_detailed(
                predictions, references, categories
            )

            results[dataset_name] = dataset_results["global"]

            # ── 简单打印 ───────────────────────
            logger.info(
                f"{dataset_name} 结果:\n"
                + CaptioningMetrics.format_results(dataset_results["global"])
            )

            # ── 保存预测 ───────────────────────
            if save_predictions:
                self._save_predictions(
                    dataset_name, predictions, references, categories,
                    image_paths, dataset_results
                )

        return results

    def _load_model(self):
        """加载模型和处理器。"""
        logger.info(f"加载模型：base={self.base_model_path}, adapter={self.model_path}")
        self.model_wrapper = QwenVLForRemoteSensing.from_pretrained(
            model_path=self.base_model_path,
            lora_adapter_path=self.model_path
            if os.path.isdir(self.model_path)
            else None,
        )

    @torch.no_grad()
    def _generate_all(
        self,
        dataset: RemoteSensingCaptionDataset,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
    ) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
        """对整个数据集逐批生成描述。

        优化点：
          - 使用批量推理加速（通过 vLLM 或 batch generation）
          - 这里使用逐样本生成 + 进度条的简单方案（兼容性最好）

        返回：
            predictions, references, categories, image_paths
        """
        predictions = []
        references = []
        categories = []
        image_paths = []

        dataset_size = len(dataset)

        for i in tqdm(range(dataset_size), desc="生成描述"):
            sample = dataset.get_sample_with_all_references(i)

            # ── 处理图像 ────────────────────
            pixel_values = sample["pixel_values"]
            prompt = "Describe this remote sensing image in detail."

            if hasattr(pixel_values, "convert"):  # PIL Image
                # 通过 apply_chat_template 构建消息（Qwen3-VL 必须走 chat template，
                # 否则 processor 不知道在哪里插入 <|vision_start|>...<|vision_end|> 图像 token）
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pixel_values},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                text = self.model_wrapper.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                processor_inputs = self.model_wrapper.processor(
                    text=[text],
                    images=[pixel_values],
                    return_tensors="pt",
                ).to(self.device)
                output_ids = self.model_wrapper.generate(
                    processor_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                # 解析：只保留新生成的 token
                prompt_len = processor_inputs["input_ids"].shape[1]
                new_tokens = output_ids[0, prompt_len:]
                caption = self.model_wrapper.decode(
                    new_tokens, skip_special_tokens=True
                )
            else:
                # 已经是张量（已过 transforms 管线，image 信息编码在 pixel_values 中）
                # 此时需要用占位符文本让 chat template 知道有图像
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "placeholder.jpg"},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                text = self.model_wrapper.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                processor_inputs = self.model_wrapper.processor(
                    text=[text],
                    images=pixel_values.unsqueeze(0) if pixel_values.dim() == 3 else pixel_values,
                    return_tensors="pt",
                ).to(self.device)
                output_ids = self.model_wrapper.generate(
                    processor_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                prompt_len = processor_inputs["input_ids"].shape[1]
                new_tokens = output_ids[0, prompt_len:]
                caption = self.model_wrapper.decode(
                    new_tokens, skip_special_tokens=True
                )

            predictions.append(caption)
            references.append(sample["captions"])
            categories.append(sample["category"])
            image_paths.append(sample["image_path"])

        return predictions, references, categories, image_paths

    def compare_with_gpt4v(
        self, gpt4v_predictions_path: str
    ) -> Dict[str, Dict[str, float]]:
        """将微调模型与 GPT-4V zero-shot 进行对比。

        参数：
            gpt4v_predictions_path: GPT-4V 的预测文件路径（JSON 格式）。
                {
                    "rsicd": {
                        "predictions": ["caption1", "caption2", ...],
                        "references": [["ref1_1", "ref1_2", ...], ...]
                    },
                    ...
                }

        返回：
            {"rsicd": {"ours_bleu_4": ..., "gpt4v_bleu_4": ..., "delta": ...}, ...}
        """
        with open(gpt4v_predictions_path, "r", encoding="utf-8") as f:
            gpt4v_data = json.load(f)

        our_results = self.evaluate(save_predictions=False)
        comparison = {}

        for dataset_name in self.dataset_names:
            if dataset_name not in gpt4v_data or dataset_name not in our_results:
                continue

            gpt4v_preds = gpt4v_data[dataset_name]["predictions"]
            gpt4v_refs = gpt4v_data[dataset_name]["references"]
            gpt4v_scores = self.metrics_calculator.compute(gpt4v_preds, gpt4v_refs)

            ours = our_results[dataset_name]
            comparison[dataset_name] = {}

            for metric, our_score in ours.items():
                gpt4v_score = gpt4v_scores.get(metric, 0.0)
                delta = our_score - gpt4v_score
                comparison[dataset_name][f"ours_{metric}"] = our_score
                comparison[dataset_name][f"gpt4v_{metric}"] = gpt4v_score
                comparison[dataset_name][f"delta_{metric}"] = round(delta, 2)

        return comparison

    # ════════════════════════════════════════════════════════════
    # 结果输出
    # ════════════════════════════════════════════════════════════

    def print_report(self, results: Dict[str, Dict[str, float]]):
        """打印格式化的评估报告。"""
        print("\n" + "=" * 70)
        print("  Qwen-VL-RS 评估报告")
        print("=" * 70)
        print(f"  模型: {self.model_path}")
        print(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)

        for dataset_name, metrics in results.items():
            print(f"\n  [{dataset_name.upper()}]")
            print(CaptioningMetrics.format_results(metrics))

        # ── 打印 Markdown 表格（方便直接粘贴到 README） ──
        print("\n" + "=" * 70)
        print("  Markdown 表格（可直接复制到 README）")
        print("=" * 70)

        headers = ["数据集"]
        all_metrics = set()
        for m in results.values():
            all_metrics.update(m.keys())
        ordered_metrics = ["bleu_4", "meteor", "rouge_l", "cider_d", "spice"]
        headers += [m for m in ordered_metrics if m in all_metrics]

        print("| " + " | ".join(headers) + " |")
        print("|" + "---|" * len(headers))

        for dataset_name in self.dataset_names:
            if dataset_name not in results:
                continue
            row = [dataset_name.upper()]
            for m in ordered_metrics:
                if m in results[dataset_name]:
                    row.append(f"{results[dataset_name][m]:.1f}")
            print("| " + " | ".join(row) + " |")

        print("=" * 70)

    def _save_predictions(
        self,
        dataset_name: str,
        predictions: List[str],
        references: List[List[str]],
        categories: List[str],
        image_paths: List[str],
        results: Dict,
    ):
        """保存预测结果和指标到 JSON 文件。"""
        output_dir = Path("experiments/evaluations")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{dataset_name}_{timestamp}.json"

        data = {
            "dataset": dataset_name,
            "model_path": self.model_path,
            "timestamp": timestamp,
            "metrics": results,
            "samples": [
                {
                    "image_path": img_path,
                    "category": cat,
                    "prediction": pred,
                    "references": refs,
                }
                for img_path, cat, pred, refs in zip(
                    image_paths, categories, predictions, references
                )
            ],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"预测结果已保存至：{output_file}")


# ════════════════════════════════════════════════════════════════
# CLI 入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen-VL-RS 模型评估")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--base_model", type=str, default=None,
                        help="基座模型路径（仅使用 LoRA adapter 时需要）")
    parser.add_argument("--datasets", type=str, default="rsicd,ucm,sydney",
                        help="要评估的数据集（逗号分隔）")
    parser.add_argument("--metrics", type=str,
                        default="bleu,meteor,rouge,cider,spice",
                        help="要计算的指标（逗号分隔）")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="推理批次大小")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    parser.add_argument("--gpt4v_predictions", type=str, default=None,
                        help="GPT-4V 预测文件路径（用于对比分析）")

    args = parser.parse_args()

    evaluator = CaptionEvaluator(
        model_path=args.checkpoint,
        base_model_path=args.base_model,
        dataset_names=args.datasets.split(","),
        metrics=args.metrics.split(","),
        device=args.device,
    )

    if args.gpt4v_predictions:
        comparison = evaluator.compare_with_gpt4v(args.gpt4v_predictions)
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
    else:
        results = evaluator.evaluate(batch_size=args.batch_size)
        evaluator.print_report(results)
