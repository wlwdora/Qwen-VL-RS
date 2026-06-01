# Qwen-VL-RS：基于视觉语言模型的遥感图像描述

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-orange.svg)](https://github.com/huggingface/peft)
[![DPO](https://img.shields.io/badge/Alignment-DPO-green.svg)](https://arxiv.org/abs/2305.18290)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**探究通用视觉语言模型在遥感图像描述领域的可行性与潜力。**

以 Qwen3-VL-2B-Instruct 为基座，通过 LoRA 高效微调（仅训练 ~5% 参数）、数据清洗与 DPO 偏好对齐，在 RSICD 基准上达到与专用架构相当的水平。全程使用单张 RTX 3060 12GB 消费级显卡完成。

核心洞察：**遥感图像描述的最大瓶颈在视觉端而非语言端。** ViT 从自然图像（地面视角）到航拍遥感（俯视视角）的域迁移远比 LLM 学习几个新词汇困难。因此视觉编码器使用 r=64 的高 rank，语言模型使用 r=32，将算力精准投向领域差距最大的地方。

---

## 📊 主要结果

### RSICD 基准对比

| 方法 | 类型 | BLEU-4 | ROUGE-L | CIDEr-D |
|------|------|--------|---------|---------|
| Transformer | 专用架构 | 35.7 | 58.2 | 155.1 |
| GCN-LSTM | 专用架构 | 33.4 | 56.5 | 138.6 |
| Up-Down | 专用架构 | 31.3 | 54.5 | 124.8 |
| **Ours (DPO 对齐)** | **通用 VLM + ~5% LoRA** | **33.2** | **55.1** | **126.5** |
| Adaptive | 专用架构 | 30.1 | 53.6 | 113.9 |
| SAT | 专用架构 | 28.0 | 49.5 | 87.0 |
| Ours (数据清洗 SFT) | 通用 VLM + ~5% LoRA | 28.8 | 51.0 | 66.0 |
| Ours (原始 SFT) | 通用 VLM + 0.8% LoRA | 21.2 | 46.6 | 1.1 |
| Qwen3-VL-2B Zero-shot | 通用 VLM | 0.8 | 3.0 | 0.0 |

> **核心发现**：通用 VLM 在 zero-shot 下完全无法完成遥感图像描述（75% 空输出），但通过针对性微调——将参数投向视觉域迁移（ViT r=64）这一核心矛盾——仅训练 ~5% 参数即可在 CIDEr 上达到 GCN-LSTM 专用架构的 91%、在 BLEU 上接近。证明了 VLM + 高效微调在遥感领域的可行性与竞争力。

---

## 🗺️ 技术路线

```
Zero-shot                SFT (原始数据)           SFT (数据清洗)          DPO 对齐
Qwen3-VL-2B              + LoRA LLM r=16          + ViT r=32, LLM r=32    + ViT r=64, 偏好训练
    │                        │                        │                      │
    ├─ 75% 空输出             ├─ 学会任务格式            ├─ 数据去重清洗           ├─ 突破 SFT 天花板
    ├─ 25% 聊天式回复         ├─ 但 CIDEr=1.1           ├─ 视觉编码器适配          ├─ 高 rank 视觉域迁移
    └─ CIDEr=0.0             └─ 只输出模板句             └─ CIDEr 1.1→66.0       └─ CIDEr 66.0→126.5
```

### 三个关键修复

1. **Collator Labels 掩码 Bug** — `_collate_pil()` 使用裸 `tokenizer()` 计算 prompt 长度，导致 270+ 个视觉 token 未被 mask 而成为训练目标，98.5% 的算力浪费在预测 padding 上。[→ error.md #9](error.md)

2. **视觉编码器未被 LoRA 覆盖** — Qwen3-VL 的 ViT 与 LLM 使用完全不同的 Linear 层命名体系（`qkv` vs `q_proj`），导致 24 层视觉编码器的 96 个 Linear 层全部冻结。加入视觉 LoRA 后 CIDEr 跃升。[→ error.md #10](error.md)

3. **多参考标注的"最小公分母"效应** — RSICD 每张图有 5 条参考描述，94.2% 存在重复。交叉熵训练下模型倾向于输出匹配最多参考的模板句，导致 CIDEr 接近零。通过 IDF 加权去重解决。[→ error.md #11](error.md)

> 📖 完整开发日志（16 个错误及解决方案）：**[error.md](error.md)**

---

## 🔬 推理效果对比

以下展示同一张遥感图像在基座模型（Zero-shot）与微调后模型（Ours DPO）上的输出对比：

| 图像 | Zero-shot (基座) | Ours (微调后) | 参考描述 |
|------|-----------------|---------------|---------|
| 🏟️ 体育场 | `assistant\nBased on the provided image, here is a detailed description...` | `a circular stadium with two crescent shaped awnings is surrounded by a pond and several buildings` | `located on a square surrounded by a pond, there is a circular stadium with two crescent shaped awnings` |
| 🅿️ 停车场 | *(空输出)* | `many cars are orderly parked in a parking lot near several buildings and green trees` | `rows of cars park neatly in this parking lot next to buildings` |
| 🌉 立交桥 | `assistant\nThis appears to be an aerial view of...` | `a complex viaduct with multiple intersecting roads is surrounded by green vegetation` | `a viaduct with several roads crossing each other is in the center of the image` |
| 🏜️ 沙漠 | *(空输出)* | `the cream colored desert has some irregular dark patterns and vertical stripes` | `some folds can be seen on the desert` |
| 🏫 密集居民区 | `assistant\nBased on the image, this is a residential area...` | `many buildings and green trees are densely packed along both sides of a road in a residential area` | `many buildings and green trees are in two sides of a road in a dense residential area` |

> **Zero-shot 问题**：基座模型无法理解图像描述任务——75% 的输入直接返回空输出，25% 返回冗长的聊天式回复而非简洁的描述语句。
>
> **微调后效果**：模型学会了直接输出遥感场景描述，能够识别地物类别（体育场/停车场/立交桥）、描述空间关系（被...环绕/沿...分布），并使用区分性词汇（crescent/creamy/orderly）。

---

## 🏗️ 模型架构

```
┌─────────────────────────────────────────────────┐
│            Qwen3-VL-2B-Instruct                  │
│                                                  │
│  ┌──────────────────┐  ┌──────────────────────┐ │
│  │ 视觉编码器 (ViT)   │  │ 语言模型 (Qwen3)       │ │
│  │ 24 blocks        │  │ 28 decoder layers    │ │
│  │                  │  │                      │ │
│  │ LoRA r=64:       │  │ LoRA r=32:           │ │
│  │  qkv, proj,      │  │  q/k/v/o_proj,       │ │
│  │  linear_fc1/2    │  │  gate/up/down_proj   │ │
│  └──────────────────┘  └──────────────────────┘ │
│                                                  │
│  可训练参数: ~95M / 2.1B (4.5%)                   │
└─────────────────────────────────────────────────┘
```

| 组件 | 模块数 | LoRA 目标层 | Rank |
|------|--------|------------|------|
| 视觉编码器 | 24× ViT blocks | `qkv`, `proj`, `linear_fc1`, `linear_fc2` | **64** |
| 语言模型 | 28× decoder layers | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | 32 |

---

## 📁 项目结构

```
Qwen-VL-RS/
├── models/
│   └── qwen_vl_rs.py              # Qwen3-VL + LoRA 封装（加载/注入/生成/合并）
├── data/
│   ├── dataset.py                 # RSICD / UCM / Sydney 数据集
│   ├── transforms.py              # 遥感图像增强（旋转/翻转/多尺度裁剪）
│   ├── collator.py                # 多模态 Data Collator
│   └── prompts.py                 # 指令模板
├── training/
│   ├── trainer.py                 # 训练循环（OneCycleLR / 早停 / 梯度累积）
│   ├── loss.py                    # Cross-Entropy / Focal Loss
│   ├── metrics.py                 # BLEU / ROUGE / CIDEr / METEOR / SPICE / CHAIR
│   └── dpo_loss.py                # DPO 损失函数 + 对数概率计算
├── evaluation/
│   ├── eval.py                    # 评估引擎（生成 + 指标计算 + 逐类别分析）
│   ├── benchmarks.py              # 多方法对比 + Markdown/LaTeX 报告
│   └── error_analysis.py          # 错误分析（幻觉率 / 地物分类 F1）
├── inference/
│   ├── infer.py                   # 单张/批量推理
│   ├── gradio_app.py              # Gradio 交互 Demo
│   └── vllm_serve.py              # vLLM 高性能推理服务
├── scripts/
│   ├── train_rsicd_r16.py         # LoRA 微调脚本
│   ├── train_dpo.py               # DPO 偏好对齐训练
│   ├── build_preference_data.py   # 偏好数据自动构建
│   ├── eval_rsicd.py              # RSICD 评估
│   ├── eval_zero_shot.py          # Zero-shot 基线
│   └── run_benchmark_full.py      # Benchmark 报告生成
├── experiments/
│   ├── benchmarks/                # Benchmark 报告 (Markdown + LaTeX)
│   └── evaluations/               # 评估结果 JSON
├── error.md                       # 开发日志：16 个错误及解决方案
└── README.md
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+, PyTorch 2.7+, CUDA 11.8+
- GPU 显存 ≥12GB（在 RTX 3060 上测试通过）
- Qwen3-VL-2B-Instruct（本地路径或 HuggingFace 下载）

### 安装

```bash
git clone https://github.com/wlwdora/Qwen-VL-RS.git
cd Qwen-VL-RS
pip install -r requirements.txt
```

### 数据准备

```bash
# 手动下载 RSICD（详见 error.md #4）
# 解压至 data/raw/RSICD_optimal/

# 预处理
python scripts/preprocess_data.py
```

### 训练

```bash
# 第一阶段：LoRA 微调
python scripts/train_rsicd_r16.py

# 第二阶段：构建偏好数据
python scripts/build_preference_data.py

# 第三阶段：DPO 偏好对齐
python scripts/train_dpo.py
```

### 推理

```python
from models.qwen_vl_rs import QwenVLForRemoteSensing
from PIL import Image

# 加载微调后模型
model = QwenVLForRemoteSensing.from_pretrained(
    model_path="Qwen/Qwen3-VL-2B-Instruct",
    lora_adapter_path="output/qwen_vl_rs_dpo/best_model",
)

# 推理
image = Image.open("scene.jpg").convert("RGB")
caption = model.predict(
    image,
    prompt="Describe this remote sensing image in detail.",
    max_new_tokens=128,
    temperature=0.7,
)
print(caption)
```

---

## 📐 评估指标说明

| 指标 | 含义 | 本项目侧重 |
|------|------|----------|
| **CIDEr-D** | TF-IDF 加权的 n-gram 共识度量，评估描述的**区分性和信息量** | ⭐ 核心指标 |
| **BLEU-4** | 4-gram 精确匹配率 + 长度惩罚 | 辅助参考 |
| **ROUGE-L** | 基于最长公共子序列的召回率 | 辅助参考 |
| METEOR | 同义词 + 词干匹配（需 Java） | Windows 不可用 |
| SPICE | 场景图语义命题评估（需 Java） | Windows 不可用 |

---

## 🔬 消融实验

| 改动 | BLEU-4 Δ | CIDEr-D Δ | 主要结论 |
|------|----------|-----------|---------|
| Collator Labels 修复 | +20.4 | +1.1 | 修复前 98.5% 算力浪费于预测 padding |
| 数据去重清洗 | +7.6 | +64.9 | 消除多参考"最小公分母"效应 |
| 视觉编码器 LoRA (r=64) | 内嵌 | 内嵌 | 航拍视角域迁移是核心矛盾，需高 rank |
| DPO 偏好对齐 | +4.4 | +60.5 | 负信号突破交叉熵天花板 |
| Prompt 中→英 | 0.0 | 0.0 | 跨语言 mismatch 非瓶颈 |

---

## 📝 开发日志

[`error.md`](error.md) 记录了本项目开发过程中遇到的 16 个错误及其解决方案，按四个阶段组织：

| 阶段 | 编号 | 关键问题 |
|------|------|---------|
| 🔧 环境与管线 | #1–6 | CUDA segfault、albumentations API 迁移、chat template |
| 🏋️ 训练调试 | #7–9 | LoRA 欠拟合、**collator labels 掩码 bug（核心修复）** |
| 📊 数据与架构 | #10–12 | **视觉编码器 LoRA 缺口**、**CIDEr "最小公分母"效应** |
| 🎯 DPO 对齐 | #13–16 | SFT 天花板分析、reward margin 崩塌、最终结果 |

每条包含现象描述、根因分析、解决方案和代码对比，既是技术参考也是系统调试方法论展示。

---

## 📚 参考文献

| 论文 | 方向 |
|------|------|
| Lu et al., *RSICD: Remote Sensing Image Captioning Dataset* (2018) | 数据集 |
| Lu et al., *Exploring Models and Data for Remote Sensing Image Captioning* (2019) | 基线方法 |
| Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) | 高效微调 |
| Rafailov et al., *Direct Preference Optimization* (NeurIPS 2023) | 偏好对齐 |
| Bai et al., *Qwen-VL: A Versatile Vision-Language Model* (2023) | 基座模型 |

---

## 👤 作者

**wlwdora** — 武汉大学电子信息学院

- **研究方向**：多模态视觉语言模型 · 遥感图像理解 · 高效微调 · 偏好对齐
- **联系方式**：2021302121165@whu.edu.cn

---

## 📄 许可证

MIT License — 学术研究与教育用途。
