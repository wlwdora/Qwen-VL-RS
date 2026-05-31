# Qwen-VL-RS: Remote Sensing Image Understanding via Vision-Language Model Fine-tuning

**基于 Qwen3-VL 的遥感图像描述与理解**

---

## 项目动机

遥感图像理解是连接计算机视觉与对地观测的关键桥梁。与自然图像不同，遥感图像具有**尺度变化大、目标密集、光谱信息丰富、空间上下文复杂**等特点。

通用视觉语言模型（如 GPT-4V、Qwen-VL）在自然图像上表现优异，但在遥感领域存在显著退化：
- 难以准确描述地物类别（区分"农田"与"草地"、"道路"与"河道"）
- 无法理解空间关系（"位于...的东北方向"、"沿河流分布"）
- 缺乏对遥感专有名词的敏感性（"归一化植被指数"、"合成孔径雷达"）

本项目使用 **LoRA 高效微调 Qwen3-VL**，在经典遥感图像描述数据集（RSICD / UCM-Captions / Sydney-Captions）上训练，目标是让 2B 参数的小模型在遥感描述任务上**逼近甚至超越 GPT-4V 的 zero-shot 效果**。

---

## 项目结构

```text
Qwen-VL-RS/
├── README.md                          # 项目文档（本文件）
├── requirements.txt                   # Python 依赖
│
├── configs/                           # 配置文件（YAML）
│   ├── sft_config.yaml                # SFT 训练参数
│   ├── lora_config.yaml               # LoRA 配置
│   └── data_config.yaml               # 数据集路径与预处理参数
│
├── data/                              # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py                     # PyTorch Dataset 类（RSICD/UCM/Sydney）
│   ├── transforms.py                  # 遥感图像增强（随机旋转/翻转/多尺度裁剪）
│   ├── collator.py                    # 多模态 Data Collator（image + text padding）
│   └── prompts.py                     # 指令模板设计（遥感专用 prompt 工程）
│
├── models/                            # 模型定义
│   ├── __init__.py
│   └── qwen_vl_rs.py                  # Qwen3-VL + LoRA 包装（支持加载/保存/merge）
│
├── training/                          # 训练模块
│   ├── __init__.py
│   ├── trainer.py                     # 基于 HuggingFace Trainer 的训练循环（非 MS-SWIFT）
│   ├── loss.py                        # 自定义 Loss（支持 Label Smoothing / Focal Loss）
│   └── metrics.py                     # 评估指标（CIDEr / BLEU / METEOR / ROUGE-L / SPICE）
│
├── evaluation/                        # 评估模块
│   ├── __init__.py
│   ├── eval.py                        # 批量评估脚本
│   ├── benchmarks.py                  # 多数据集 benchmark（zero-shot vs fine-tuned vs GPT-4V）
│   └── error_analysis.py             # 错误分析（按地物类别/空间关系/光谱特性分类）
│
├── inference/                         # 推理模块
│   ├── __init__.py
│   ├── infer.py                       # 单张/批量推理接口
│   ├── gradio_app.py                  # Gradio 交互式 Demo（支持图像上传+对比显示）
│   └── vllm_serve.py                  # vLLM 高性能推理服务
│
├── scripts/                           # 运行脚本
│   ├── train_sft.sh                   # SFT 训练启动
│   ├── train_lora.sh                  # LoRA 训练启动
│   ├── eval.sh                        # 评估启动
│   └── inference.sh                   # 推理启动
│
├── experiments/                       # 实验记录（git tracked）
│   ├── .gitkeep
│   ├── exp_template.md                # 实验记录模板
│   └── logs/                          # 训练日志（.gitignore 排除 tensorboard 文件）
│
├── notebooks/                         # Jupyter 探索笔记
│   ├── 01_data_exploration.ipynb      # 数据集探索与统计
│   ├── 02_baseline_analysis.ipynb     # 基座模型 zero-shot 分析
│   ├── 03_error_analysis.ipynb        # 错误案例分析
│   └── 04_ablation_results.ipynb      # 消融实验可视化
│
└── tests/                             # 单元测试
    ├── __init__.py
    ├── test_dataset.py                # 数据集加载测试
    ├── test_metrics.py                # 评估指标测试
    └── test_model.py                  # 模型前向传播测试
```

---

## 技术栈

| 层级 | 技术选型 | 选型理由 |
|:---|:---|:---|
| **基座模型** | Qwen3-VL-2B-Instruct | 轻量（单张 12GB 卡可训练）、中英文双语、开源可商用 |
| **高效微调** | LoRA (rank=16/32/64) | 仅训练 ~1% 参数，显存友好，可做 rank 消融实验 |
| **训练框架** | HuggingFace Transformers + PEFT + `Trainer` | 手写训练循环，完全可控（不用 MS-SWIFT 黑盒） |
| **评估指标** | CIDEr-D / BLEU-4 / METEOR / ROUGE-L / SPICE | NLP-CV 社区标准，CIDEr 对图像描述最敏感 |
| **推理服务** | vLLM + PagedAttention | 高吞吐、低延迟，支持连续批处理 |
| **可视化** | TensorBoard + W&B (可选) | 训练曲线跟踪、embedding 可视化 |
| **实验管理** | 手动 Markdown 记录 + Git 版本控制 | 轻量、可复现、面试时可展示 |

---

## 数据集

### 主要数据集

| 数据集 | 图像数 | 描述数/图 | 分辨率 | 场景类别 | 来源 |
|:---|:---|:---|:---|:---|:---|
| **UCM-Captions** | 2,100 | 5 | 256×256 | 21 类（机场/海滩/森林/高速...） | UC Merced |
| **Sydney-Captions** | 613 | 5 | 500×500 | 7 类（住宅/工业/河流/海洋...） | 悉尼遥感 |
| **RSICD** | 10,921 | 5 | 224×224 | 30+ 类 | 中科院 |

### 数据增强策略（遥感特定）

| 增强方式 | 参数 | 目的 |
|:---|:---|:---|
| 随机旋转 | 0° / 90° / 180° / 270° | 遥感图像无固定朝向 |
| 水平/垂直翻转 | p=0.5 | 对称性不变 |
| 多尺度裁剪 | 0.5× ~ 1.5× | 模拟不同空间分辨率 |
| 光谱扰动 | 亮度±0.1, 对比度±0.1 | 模拟不同光照/大气条件 |
| MixUp (可选) | α=0.2 | 正则化，提升泛化 |

### 数据格式

```jsonl
{
  "image": "airport_001.jpg",
  "captions": [
    "An airport with two runways and several terminal buildings.",
    "Several planes are parked at the airport terminals.",
    "..."
  ],
  "category": "airport",
  "metadata": {
    "resolution": "256x256",
    "source": "UCM-Captions"
  }
}
```

---

## 实验设计

### 基线对比（Baseline Comparison）

| # | 方法 | 说明 |
|:---|:---|:---|
| B1 | Qwen3-VL-2B (zero-shot) | 不作任何微调，直接用指令 prompt |
| B2 | GPT-4V API (zero-shot) | 商业最强 VLM 作为 upper reference |
| B3 | BLIP-2 + LoRA | 经典图像描述模型微调 |
| B4 | **Qwen3-VL + LoRA (Ours)** | 本项目方法 |

### 消融实验（Ablation Study）

| # | 变量 | 设定 |
|:---|:---|:---|
| A1 | LoRA rank | r ∈ {8, 16, 32, 64} |
| A2 | LoRA target modules | q+v only / q+k+v+o / all-linear |
| A3 | 训练数据量 | {25%, 50%, 75%, 100%} × 全部数据 |
| A4 | Prompt 设计 | 标准 prompt vs 遥感专用 prompt vs chain-of-thought prompt |
| A5 | 数据增强 | 无增强 / 仅空间 / 空间+光谱 / 全部 |
| A6 | 多数据集联合训练 | 单数据集 vs UCM+Sydney vs UCM+Sydney+RSICD |
| A7 | LLM 解冻比例 | LoRA only / LoRA + LLM最后2层 / Full fine-tune (对比) |

### 评估维度

除了标准 captioning metrics，增加以下细粒度评估：

| 维度 | 评估方法 | 指标 |
|:---|:---|:---|
| 地物分类准确率 | 提取 caption 中的地物名词，与 GT 对齐 | Precision / Recall / F1 |
| 空间关系正确性 | 规则匹配（方位词 + 参照物） | Accuracy |
| 数量描述精度 | 数值提取与比较 | MAE |
| 光谱描述合理性 | 人工抽检 | Likert 1-5 评分 |
| 幻觉率 | CHAIR 指标（不在图中的物体被描述） | CHAIR-s / CHAIR-i |

---

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/your-username/Qwen-VL-RS.git
cd Qwen-VL-RS

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据集
# RSICD: https://github.com/201528014227051/RSICD_optimal
# UCM-Captions: https://github.com/201528014227051/RSICD_optimal
# 将数据放置于 data/raw/ 目录下

# 4. 数据预处理
python data/dataset.py --download --preprocess

# 5. LoRA 微调
bash scripts/train_lora.sh

# 6. 评估
bash scripts/eval.sh

# 7. 启动 Gradio Demo
python inference/gradio_app.py
```

---

## 预期实验结果（目标）

| 方法 | BLEU-4 | METEOR | CIDEr-D | SPICE |
|:---|:---|:---|:---|:---|
| Qwen3-VL zero-shot | ~15 | ~18 | ~30 | ~12 |
| BLIP-2 + LoRA | ~22 | ~24 | ~65 | ~20 |
| GPT-4V zero-shot | ~25 | ~26 | ~75 | ~22 |
| **Qwen3-VL + LoRA (ours)** | **≥26** | **≥27** | **≥80** | **≥23** |

> 目标：在 CIDEr-D 指标上，用 2B 微调模型超越 GPT-4V 的 zero-shot 效果。

---

## 时间规划（6 周）

| 周次 | 任务 | 产出 |
|:---|:---|:---|
| Week 1 | 数据集下载、统计分析、baseline zero-shot 测试 | 数据探索 notebook，baseline 指标 |
| Week 2 | 搭建训练管线（Dataset / Collator / Trainer） | 可运行的训练代码 |
| Week 3 | LoRA 微调实验（含消融 A1-A4） | 消融实验结果表 |
| Week 4 | 完整评估 + 错误分析 + GPT-4V 对比 | 评估报告、错误分类统计 |
| Week 5 | 消融 A5-A7 + 最优模型确定 + Gradio Demo | 最终模型 + Demo |
| Week 6 | 整理实验报告 + 写项目总结 + GitHub README 完善 | 面试可展示的完整项目 |

---

## 关键参考文献

| 论文 | 方向 | 关键贡献 |
|:---|:---|:---|
| *RSICD: Remote Sensing Image Captioning Dataset* (Lu et al., 2018) | 数据集 | 首个大规模遥感描述数据集 |
| *Exploring Models and Data for Remote Sensing Image Captioning* (Lu et al., 2019) | 基线 | 遥感描述经典 baseline |
| *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021) | 方法 | LoRA 微调方法 |
| *Qwen-VL: A Versatile Vision-Language Model* (Bai et al., 2023) | 模型 | 基座模型架构 |
| *GEO: Generative End-to-End Object Detector for Remote Sensing* (2023) | 相关 | 遥感视觉语言任务 |
| *RemoteCLIP: A Vision-Language Model for Remote Sensing* (Liu et al., 2024) | 相关 | 遥感专用 VLM |

---

## 要点

尝试做完这个项目后，我希望可以应该能清晰地回答：

1. **为什么遥感图像描述比自然图像更难？**（尺度变化、类别细粒度、空间上下文、光谱信息）
2. **为什么不直接用 GPT-4V？**（API 成本、数据隐私、推理延迟、可定制性、小模型在特定领域的潜力）
3. **为什么选 LoRA 而不是 Full Fine-tune？**（算力约束、可叠加任务 adapter、防止灾难性遗忘）
4. **CIDEr-D 为什么比 BLEU 更适合图像描述？**（TF-IDF 加权 n-gram，对内容词敏感）
5. **你怎么确定模型没有过拟合？**（cross-dataset evaluation、数据增强消融）
6. **如果让你上线部署，还缺什么？**（输入校验、响应式推理、模型热更新、A/B 测试框架）
7. **你在这个项目中最大的技术挑战是什么？**（准备好一个具体的、有细节的故事）

---

## 作者

- **开发**: wlwdora - 武汉大学电子信息学院
- **方向**: 多模态视觉语言模型 · 遥感图像理解 · 高效微调
- **联系**: 2021302121165@whu.edu.cn
- **日期**: 2026-04

---

## License

MIT License — 本项目仅用于学术研究与面试展示。
