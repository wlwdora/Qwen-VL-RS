# Qwen-VL-RS: Remote Sensing Image Captioning via Vision-Language Model Fine-tuning

**基于 Qwen3-VL-2B 的遥感图像描述 —— 从 zero-shot 失败到超越文献基线**

---

## 项目概览

使用 LoRA 高效微调 + 数据工程 + DPO 偏好对齐，将 Qwen3-VL-2B-Instruct 从**完全无法做遥感图像描述**（zero-shot BLEU-4=0.8, CIDEr=0.0）提升到**可用水平**（BLEU-4=29.8, CIDEr=92.3），在 RSICD 基准上接近专用架构方法。

**核心贡献**：
- 定位并修复了导致 98.5% 算力浪费的 collator labels 掩码 bug
- 发现 visual encoder 未被 LoRA 覆盖的架构问题并修复
- 提出基于 IDF 区分性评分的自动偏好数据构建方法用于 DPO 对齐
- 完整记录了 16 个开发/调试错误的解决方案

---

## 最终 Benchmark（RSICD）

| 方法 | BLEU-4 | ROUGE-L | CIDEr-D |
|------|--------|---------|---------|
| Transformer (文献 SOTA) | 35.7 | 58.2 | 155.1 |
| GCN-LSTM | 33.4 | 56.5 | 138.6 |
| Up-Down | 31.3 | 54.5 | 124.8 |
| Adaptive | 30.1 | 53.6 | 113.9 |
| SAT | 28.0 | 49.5 | 87.0 |
| **Ours v3 (DPO 对齐)** | **29.8** | **51.7** | **92.3** |
| Ours v2 (数据清洗 + visual LoRA SFT) | 27.8 | 50.3 | 54.0 |
| Ours v1 (collator 修复, 原始数据 SFT) | 21.2 | 46.6 | 1.1 |
| Zero-shot (基座模型) | 0.8 | 3.0 | 0.0 |

---

## 技术演进路线（v0 → v3）

```
v0: 基座模型 zero-shot
    → 75% 空输出, 25% 聊天式回复, 完全不可用

v1: LoRA r=16, collator bug 修复
    → 学会任务格式, BLEU-4=21.2, 但 CIDEr=1.1
    → 根因：原始 RSICD 5 条参考中大量重复模板，训练信号把模型推向"安全通用表达"

v2: 数据去重清洗 + visual encoder LoRA r=16 + LLM LoRA r=32
    → 数据清洗消除"最小公分母"效应，visual LoRA 让模型看清遥感特征
    → CIDEr 1.1→54.0, BLEU-4 21.2→27.8
    → 但 SFT 有天花板：交叉熵只给"正面信号"，无法教会模型"远离坏的输出"

v3: DPO 偏好对齐
    → 直接对比 chosen(详细) vs rejected(模板)，给模型"负信号"
    → CIDEr 54.0→92.3, 超过 SAT (87.0)，突破 SFT 天花板
```

### 关键 Bug 修复

1. **Collator labels 掩码 bug**（error 9）：`_collate_pil()` 用裸 `tokenizer()` 计算 prompt_length=19，但 `processor()` 将视觉 token 展开为 270+ 个。labels 只 mask 前 19 个，导致 270 个 `<|image_pad|>` token 成为训练目标——模型 98.5% 算力浪费在预测 padding token。

2. **Visual encoder 未被 LoRA 覆盖**（error 10）：Qwen3-VL 的 ViT 使用 `qkv/proj/linear_fc1/linear_fc2` 层名，与 LLM 的 `q_proj/k_proj/v_proj/gate/up/down_proj` 完全不同。当前 target_modules 全部匹配到 LLM 侧，24 层 visual encoder 的 96 个 Linear 层全部冻结。

3. **多参考标注的"最小公分母"效应**（error 11）：RSICD 94.2% 的样本存在重复参考。当一张图有 3 条 "many buildings and trees" 模板 + 2 条 "circular truncated cone with rhombus lawn" 详细描述时，交叉熵训练天然倾向于模板——因为模板能匹配更多参考，loss 更低。**这是 v1 CIDEr=1.1 的根因**，不是模型没学会，而是训练信号本身就是错的。

### 数据清洗策略

从 RSICD 原始 5 条参考中自动筛选：
1. **去重**：合并完全重复的参考
2. **区分性评分**：全局 IDF 加权 + 长度惩罚，量化每条 caption 的信息量
3. **保留 top-2**：只保留最有区分性的 1-2 条作为训练参考

效果：模板占比 40%→15%，区分性词汇（"crescent", "awnings", "bleachers"）不再被淹没。

### DPO 偏好对齐

数据清洗后 SFT 模型的 CIDEr=54.0 已经可用，但交叉熵训练有根本局限——它只告诉模型"什么是对的"，不告诉"什么是错的"。当 "colorful bleachers" 和 "green trees" 在 stadium 上下文中概率相近时，SFT 无法推动模型选择更区分性的表达。

DPO 直接解决这个问题：
- 从原始 RSICD 中构建 5781 个偏好对：chosen=最具区分性的参考，rejected=最模板化的参考
- 训练目标：最大化 `P(chosen)/P(rejected)` 相对于冻结 SFT 模型的比值
- CIDEr 从 54.0 提升到 92.3（+71%），超过 SAT 文献方法 (87.0)

---

## 项目结构

```text
Qwen-VL-RS/
├── README.md
├── error.md                          # 16 个开发错误及解决方案（面试重点）
├── requirements.txt
│
├── configs/                          # 训练/LoRA/数据配置
│   ├── sft_config.yaml
│   ├── lora_config.yaml
│   └── data_config.yaml
│
├── data/
│   ├── dataset.py                    # RSICD / UCM / Sydney Dataset 类
│   ├── transforms.py                 # 遥感图像增强（旋转/翻转/多尺度）
│   ├── collator.py                   # 多模态 Data Collator（含 bug 修复）
│   ├── prompts.py                    # Prompt 模板
│   └── processed/                    # 预处理后的数据
│       ├── rsicd.jsonl               # 原始 RSICD
│       └── rsicd_dpo_pairs.jsonl     # DPO 偏好对 (5781 对)
│
├── models/
│   └── qwen_vl_rs.py                 # Qwen3-VL + LoRA 封装（加载/注入/生成/合并）
│
├── training/
│   ├── trainer.py                    # 训练循环
│   ├── loss.py                       # Cross-entropy / Focal loss
│   ├── metrics.py                    # BLEU / ROUGE / CIDEr / METEOR / SPICE / CHAIR
│   └── dpo_loss.py                   # DPO loss + log-prob 计算
│
├── evaluation/
│   ├── eval.py                       # 评估引擎（生成 + 指标 + 逐类别分析）
│   ├── benchmarks.py                 # 多方法 benchmark 对比 + Markdown/LaTeX 报告
│   └── error_analysis.py            # 错误分析（幻觉率、地物分类细粒度）
│
├── inference/
│   ├── infer.py                      # 单张/批量推理
│   ├── gradio_app.py                 # Gradio 交互 Demo
│   └── vllm_serve.py                 # vLLM 高性能推理
│
├── scripts/
│   ├── train_rsicd_r16.py           # v1: LoRA r=16 训练
│   ├── eval_rsicd.py                # RSICD 评估
│   ├── eval_zero_shot.py            # Zero-shot baseline 评测
│   ├── build_preference_data.py     # DPO 偏好数据构建
│   ├── train_dpo.py                 # DPO 训练
│   ├── run_benchmark_full.py        # Benchmark 报告生成
│   ├── preprocess_data.py           # 数据预处理
│   └── smoke_test.py                # 管线冒烟测试
│
├── experiments/
│   ├── benchmarks/                   # Benchmark 报告 (Markdown + LaTeX)
│   └── evaluations/                  # 评估结果 JSON
│
├── output/                           # 模型 checkpoint
│   ├── qwen_vl_rs_lora_r16_fixed/   # v1 最佳模型
│   ├── qwen_vl_rs_lora_v2_dedup/    # v2 最佳模型
│   └── qwen_vl_rs_dpo/              # v3 DPO 模型
│
└── tests/
    ├── test_dataset.py
    ├── test_metrics.py
    └── test_model.py
```

---

## 技术栈

| 层级 | 技术 | 选型理由 |
|------|------|---------|
| 基座模型 | Qwen3-VL-2B-Instruct | 2.1B 参数，RTX 3060 12GB 可训，中英双语 |
| 高效微调 | LoRA (PEFT) | 仅训练 0.8%~3% 参数，支持 visual + LLM 分层 rank |
| 偏好对齐 | DPO (Direct Preference Optimization) | 无需 reward model，直接从偏好对优化 |
| 训练框架 | PyTorch + Transformers + 手写训练循环 | 完全可控，便于调试 |
| 评估指标 | pycocoevalcap (BLEU/ROUGE/CIDEr/METEOR/SPICE) | COCO 标准协议 |
| 可视化 | TensorBoard | 训练曲线实时监控 |

---

## 关键实验结果

### Zero-shot 分析（基座模型 vs 微调后）

基座 Qwen3-VL-2B **无法完成遥感图像描述任务**：
- 75% 的输入产生空输出
- 25% 产生冗长的聊天式回复（"Based on the provided image, here is a detailed description..."）
- 零条预测是直接可用的图像描述

LoRA 微调从根本上改变了模型行为——不仅提升质量，更是**学会了任务格式**。

### 逐类别分析（v2）

| 类别 | BLEU-4 | 特征 |
|------|--------|------|
| parking | 44.2 | 视觉模式高度规整，描述固定 |
| denseresidential | 41.9 | 居民区模式较一致 |
| forest | 31.2 | 植被覆盖，描述相对简单 |
| bridge | 31.0 | 线性结构较易识别 |
| square | 7.4 | 布局多变，需复杂空间推理 |
| center | 7.9 | 城市化密集区域，多目标多关系 |

复杂类别（square/center/beach）的 BLEU-4 仅为简单类别（parking）的 1/6，反映出模型在需要复杂空间推理和多目标关系描述时仍有明显短板。

### 消融分析

| 变动 | BLEU-4 Δ | CIDEr-D Δ | 说明 |
|------|----------|-----------|------|
| Collator bug 修复 | +20.4 | +1.1 | 修复前在预测 padding，修复后才真正开始学习 |
| 数据去重清洗 | +6.6 | +52.9 | **CIDEr 最大单次增益**：消除"最小公分母"，模板占比 40%→15% |
| Visual encoder LoRA | 内嵌 | 内嵌 | 与数据清洗同时应用，让视觉特征匹配清洗后的高质量参考 |
| DPO 偏好对齐 | +2.0 | +38.3 | 突破 SFT 天花板，教会模型主动选择区分性词汇 |
| Prompt 中→英 | +0.0 | +0.0 | 跨语言 mismatch 不是瓶颈（error 8） |

---

## 面试要点（WHY 问题）

### 1. 为什么遥感图像描述比自然图像难？
- 航拍俯视视角 vs 地面平视——视觉模式完全不同
- 尺度变化大（1m~30m 分辨率），同一地物在不同分辨率下外观差异巨大
- 类别细粒度：区分"草地/农田/高尔夫球场"、"道路/河道/铁路"
- 空间关系复杂：需要描述"位于...东北方向"、"沿河流分布"等

### 2. 为什么不用 GPT-4V API？
- 成本：API 调用不可持续
- 可控性：无法修改模型行为，无法针对性优化
- 学术价值：证明小模型 + 微调可以在特定领域匹敌/超越通用大模型
- 部署：本地推理无延迟、无隐私问题

### 3. 为什么选 LoRA 而不是全量微调？
- RTX 3060 12GB 无法装下 2.1B 模型的全量梯度 + 优化器状态
- LoRA adapter 仅 ~70MB，易于版本管理和多任务切换
- 保留基座模型的通用能力，避免灾难性遗忘
- 显存约束下的最优选择

### 4. CIDEr-D 为什么比 BLEU 更适合图像描述？
- BLEU 只看 n-gram 精确匹配，忽略了词的语义重要性
- CIDEr 用 TF-IDF 加权：高频通用词（"many", "area"）权重低，稀有区分词（"crescent", "awnings"）权重高
- 实验中：v1 CIDEr=1.1 vs BLEU-4=21.2，说明模型虽然在措辞上接近参考（BLEU 还行），但完全没有生成区分性内容（CIDEr 极低）

### 5. 如何确定没有过拟合？
- train/eval loss gap 监控：v1 阶段 gap 仅 0.01（欠拟合，非过拟合）
- 逐类别 BLEU 方差极大（parking 44.2 vs square 7.4）→ 模型在简单类别上表现好，复杂类别差 → 不是过拟合（过拟合应所有类别都差）
- 跨 split 评估：train/val/test 的 loss 趋势一致

### 6. 如果上线部署，还缺什么？
- 输入校验：图像分辨率/格式/通道数检查
- 推理优化：vLLM PagedAttention / FlashAttention-2
- 模型热更新：LoRA adapter 热切换（无需重启服务）
- 监控：生成质量漂移检测、异常输出告警
- A/B 测试框架：对比不同 prompt / LoRA rank 的效果

### 7. 最大技术挑战？
定位 collator bug 的过程。训练了 3 个 epoch × 3 轮实验（r=8 / r=16 中/英文），loss 始终卡在 7.34。排查了 prompt 语言（error 8）、LoRA rank（error 7），最终解码 labels 发现 270 个 `<|image_pad|>` 全是训练目标——98.5% 的算力都在预测 padding token。这个 bug 的隐蔽之处在于代码能跑通、loss 会下降、有指标输出，一切看起来"正常"，但实际学习信号完全错误。

---

## 开发环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 3060 12GB |
| CUDA | 11.8 |
| Python | 3.10 (Anaconda) |
| PyTorch | 2.7.1 |
| 基座模型 | Qwen3-VL-2B-Instruct (本地 `D:/Qwen/`) |
| 数据集 | RSICD (10,921 张, 30+ 类, 5 参考/图) |

---

## 快速开始

```bash
# 1. 安装
pip install -r requirements.txt

# 2. 数据预处理
python scripts/preprocess_data.py

# 3. LoRA 微调（v1）
python scripts/train_rsicd_r16.py

# 4. 数据去重 + 构建偏好数据
python scripts/build_preference_data.py

# 5. DPO 训练（v3）
python scripts/train_dpo.py

# 6. 评估
python scripts/eval_rsicd.py

# 7. Benchmark 报告
python scripts/run_benchmark_full.py
```

---

## 参考文献

| 论文 | 方向 | 贡献 |
|------|------|------|
| *RSICD: Remote Sensing Image Captioning Dataset* (Lu et al., 2018) | 数据集 | 首个大规模遥感描述数据集 |
| *Exploring Models and Data for Remote Sensing Image Captioning* (Lu et al., 2019) | 基线 | 遥感描述经典 baseline |
| *LoRA: Low-Rank Adaptation* (Hu et al., 2021) | 方法 | LoRA 微调 |
| *Direct Preference Optimization* (Rafailov et al., 2023) | 对齐 | DPO 偏好训练 |
| *Qwen-VL: A Versatile Vision-Language Model* (Bai et al., 2023) | 模型 | 基座模型架构 |

---

## 作者

- **wlwdora** — 武汉大学电子信息学院
- **方向**: 多模态视觉语言模型 · 遥感图像理解 · 高效微调 · 偏好对齐
- **联系**: 2021302121165@whu.edu.cn
- **日期**: 2026-06

---

## License

MIT License — 学术研究与面试展示用途。
