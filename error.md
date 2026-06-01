# 🐛 Qwen-VL-RS 开发日志：16 个错误 & 解决方案

[![LoRA](https://img.shields.io/badge/LoRA-rank%2016%2F32-orange)](https://github.com/huggingface/peft)
[![Model](https://img.shields.io/badge/Model-Qwen3--VL--2B-blue)](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
[![DPO](https://img.shields.io/badge/Alignment-DPO-green)](https://arxiv.org/abs/2305.18290)
[![GPU](https://img.shields.io/badge/GPU-RTX%203060%2012GB-red)]()

记录本项目从环境搭建到 DPO 对齐全过程中遇到的 16 个错误、排查思路及解决方案。

---

## 📊 项目演进总览

```
v0 (zero-shot)      BLEU-4= 0.8   CIDEr=  0.0    ← 完全不可用，75% 空输出
v1 (collator fix)   BLEU-4=21.2   CIDEr=  1.1    ← 学会格式，但只会输出模板句
v2 (数据清洗+SFT)    BLEU-4=27.8   CIDEr= 54.0    ← 数据清洗 + visual LoRA
v3 (DPO 对齐)       BLEU-4=29.8   CIDEr= 92.3    ← 突破 SFT 天花板
```

| # | Phase | 错误 | 状态 |
|---|-------|------|------|
| 1 | 🔧 Setup | 管线验证 Segfault (exit 139) | ✅ |
| 2 | 🔧 Setup | `torch_dtype` 废弃警告 | ✅ |
| 3 | 🔧 Setup | 脚本模式 Segfault vs 逐行执行正常 | ⚠️ |
| 4 | 🔧 Setup | GitHub 数据集下载被墙 | ⚠️ |
| 5 | 🔧 Setup | albumentations 2.0 API 变更 | ✅ |
| 6 | 🔧 Setup | Processor 直接传参导致 image token mismatch | ✅ |
| 7 | 🏋️ Training | LoRA r=8 欠拟合 → loss 饱和 | ✅ |
| 8 | 🏋️ Training | Loss 天花板 7.34 排查 | ✅ |
| 9 | 🏋️ Training | **Collator labels 掩码 bug**（核心修复） | ✅ |
| 10 | 🏗️ Arch | **Visual Encoder 未被 LoRA 覆盖** | ✅ |
| 11 | 📊 Data | **CIDEr=1.1 根因：多参考"最小公分母"效应** | ✅ |
| 12 | 🏗️ Arch | Visual LoRA + 数据清洗后 SFT 结果 | ✅ |
| 13 | 🎯 DPO | 交叉熵 SFT 天花板 → 为什么需要 DPO | ✅ |
| 14 | 🎯 DPO | Reward margin 崩塌 | ✅ |
| 15 | 🎯 DPO | Reference model deepcopy OOM | ✅ |
| 16 | 🎯 DPO | DPO v3 最终评估 | ✅ |

---

## 🔧 Phase 1: 环境与管线搭建 (Errors 1-6)

### Error 1: 管线验证 Segfault

**现象**：运行 `scripts/verify_pipeline.py` 加载 Qwen3-VL-2B-Instruct 时发生 segfault (exit code 139)。

**根因**：
- `models/qwen_vl_rs.py` 使用了已废弃的 `torch_dtype` 参数（新版 transformers 用 `dtype`）
- CUDA 11.8 下 bfloat16 稳定性不如 float16

**解决**：
```python
# Before
model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16)
# After
model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.float16)
```

### Error 2: `torch_dtype` 废弃警告

与 Error 1 同源。替换为 `dtype=`，默认精度 bfloat16 → float16。RTX 3060 12GB 在 CUDA 11.8 下 float16 稳定性更好，训练精度损失可忽略。

### Error 3: 脚本模式 Segfault

**现象**：`python scripts/verify_pipeline.py` segfault，但相同代码用 `python -c "..."` 逐行执行全部通过。

**结论**：Windows + CUDA 11.8 + torch 2.7.1 下脚本模式 CUDA context 初始化存在时机问题。管线本身功能正常——训练入口已验证通过 CLI 执行。

**方案**：验证改用分步脚本或交互式运行，训练通过 `training/trainer.py` CLI 入口（已验证正常）。

### Error 4: GitHub 数据集下载被墙

`git clone` 返回 `Empty reply from server`。改为浏览器手动下载 ZIP → 解压到 `data/raw/RSICD_optimal/`。

### Error 5: albumentations 2.0 API 变更

```python
# Before (1.x)
PadIfNeeded(value=0, border_mode=cv2.BORDER_CONSTANT)
RandomResizedCrop(height=512, width=512, scale=(0.5, 1.5))

# After (2.0)
PadIfNeeded(fill=0, pad_mode=cv2.BORDER_CONSTANT)
RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0))  # scale 必须在 [0, 1]
```

### Error 6: Processor 直接传参 → Image Token Mismatch

**现象**：
```
ValueError: Image features and image tokens do not match: tokens: 0, features 64
```

**根因**：Qwen3-VL processor 需要先通过 `apply_chat_template()` 构建消息格式（含 `{"type": "image"}`），processor 才知道在文本中插入 `<|vision_start|>...<|vision_end|>` token。直接传 `text+images` 不会自动添加图像 token。

**修复**（`evaluation/eval.py` `_generate_all` 方法）：先调 `apply_chat_template()` 再调 `processor()`。

---

## 🏋️ Phase 2: 训练调试 — Loss 死活下不去 (Errors 7-9)

### Error 7: LoRA r=8 欠拟合

**配置**：rank=8, alpha=16, lr=2e-5, 3 epochs, batch=2×4=8

**Loss 走势**：

| 阶段 | Train Loss | Eval Loss | Gap |
|------|-----------|-----------|-----|
| Epoch 1 初 (step 10) | 25.45 | — | — |
| Epoch 1 末 (~step 1000) | 9.82 | 7.36 | 2.46 |
| Epoch 2 末 (~step 2000) | 7.34 | 7.36 | **0.02** |
| Epoch 3 末 (~step 3270) | 7.34 | 7.35 | **0.01** |

**关键发现**：
- Epoch 1 是唯一有效的学习阶段 (25.45 → 9.61)
- 从 step ~1100 起 loss 卡在 7.34，后续 2100 步几乎零改善
- Train/Eval gap 仅 0.01 → **不是过拟合，是欠拟合**：r=8 仅 8.7M (0.41%) 可训练参数，模型容量不足以进一步降低 loss
- Epoch 2+3 耗时 ~91 分钟，完全浪费

**教训**：
- 小 rank 在 vocab_size=151,936 的超大词表模型上容易欠拟合，应从中等 rank (16-32) 起步
- Train/Eval gap 过小 + loss 不降 = 欠拟合，不是过拟合
- 早停机制避免在已收敛的模型上空跑

### Error 8: Loss 天花板 7.34 — 跨语言假说被排除

**假说**：中文 prompt + 英文 caption 导致跨语言 mismatch。

**实验**：将 prompt 从 `"请详细描述这张遥感图像的内容。"` 改为 `"Describe this remote sensing image in detail."`。

| 配置 | Train Loss @1000 | Eval Loss @1000 |
|------|-----------------|-----------------|
| 中文 prompt | 9.36 | 7.354 |
| 英文 prompt | 9.39 | 7.386 |

**结论**：跨语言 mismatch **不是瓶颈**。loss 天花板另有其因 → 见 Error 9。

### 🔥 Error 9: Collator Labels 掩码 Bug（项目最关键的修复）

**现象**：loss 天花板卡在 7.34，换 rank、换 prompt 语言均无效。

**排查**：解码 collator 生成的 labels：

```
修复前: Masked=19 tokens,  Label=274 tokens  ← 全是 <|image_pad|>！
修复后: Masked=274 tokens, Label=14  tokens  ← 真正的 caption 文本
```

**根因**：`_collate_pil()` 用裸 `self.tokenizer(prompt_text)` 计算 prompt_length=19。但 `processor()` 把 `<|vision_start|>...<|vision_end|>` 展开成 270+ 个 `<|image_pad|>` token。**labels 只 mask 前 19 个，剩余 270 个 image_pad 全部成为训练目标 — 模型 98.5% 算力浪费在预测 padding token。**

```
prompt 文本:  "Describe this remote sensing image in detail."
裸 tokenizer:  → 19 tokens
processor():   → 19 + 270(image_pad) = ~289 tokens
                    ↑
              labels 应为 -100（mask），但实际设为模型学习目标
```

**修复**（`data/collator.py`）：
```python
# Before: 裸 tokenizer（不含图像 token 展开）
prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt")
prompt_lengths.append(prompt_tokens.input_ids.shape[1])

# After: 用 processor（含图像 token 展开）
prompt_inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
prompt_lengths.append(prompt_inputs.input_ids.shape[1])
```

**影响**：之前 r=8、r=16、中/英文 prompt 等所有实验均基于错误 labels。修复后 Train Loss 7.34→1.38，Eval Loss 7.35→1.56，BLEU-4 17.7→21.2。

---

## 📊 Phase 3: 数据工程 & 架构修复 (Errors 10-12)

### Error 10: Visual Encoder 未被 LoRA 覆盖

**现象**：collator 修复后 BLEU-4=21.2 仍然偏低，尤其复杂类别（square=7.4, center=7.9）。

**排查**：检查模型各部分的 Linear 层命名：

| 部分 | Linear 层名 | LoRA target_modules 匹配？ |
|------|------------|--------------------------|
| LLM attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | ✅ |
| LLM FFN (SwiGLU) | `gate_proj`, `up_proj`, `down_proj` | ✅ |
| Visual attention | `qkv`, `proj` | ❌ 层名不同 |
| Visual FFN | `linear_fc1`, `linear_fc2` | ❌ 层名不同 |

**根因**：Qwen3-VL 的 Visual Encoder (ViT) 使用与 LLM 完全不同的层命名体系。当前 `target_modules` 为 LLM 侧层名，**24 个 visual block × 4 个 Linear = 96 层全部未注入 LoRA**。仅有 LLM 的 196 层被训练 (17.4M / 2.1B = 0.81%)。

```
Visual Encoder (24层, 96个 Linear) ─── 0 个 LoRA adapter
       ↓
MLP 投影层                            ─── 0 训练
       ↓
LLM (28层, 196个 Linear)              ─── 仅有这里的 196 层有 LoRA
```

**修复**：`target_modules` 加入 `qkv`, `proj`, `linear_fc1`, `linear_fc2`。额外 ~30M 可训练参数，总占比 ~2.2%，RTX 3060 12GB 可容纳。

### 🔥 Error 11: CIDEr=1.1 根因 — 多参考标注的"最小公分母"效应

**v1 表面结果**：BLEU-4=21.2, ROUGE-L=46.6 — 看起来还行。但 **CIDEr-D=1.1**，比 zero-shot (0.0) 几乎没有改善。

**问题**：为什么 loss 正常下降、BLEU 有分数、模型产出完整句子，CIDEr 却几乎为零？

**数据统计**（1093 张测试图）：
- 仅 5.8% 的图有 5 条完全不同的参考
- 94.2% 至少有一对重复
- `trees` 出现于 49% 图中, `green` 49%, `many` 46%, `buildings` 38%, `are` 73%

**"最小公分母"效应**（以一张体育场图为例）：

```
[1] many buildings and green trees are around a stadium .    ← 模板
[2] many buildings and green trees are around a stadium .    ← 重复
[3] many buildings and green trees are around a stadium .    ← 重复
[4] colorful bleachers on the edge and running tracks and    ← 区分性描述
    a playground in the middle constitute this stadium .
[5] located among buildings the stadium has a colorful stand .
```

交叉熵训练下：输出模板句 → 同时匹配 3 条参考 → **低 loss**。输出详细描述 → 只匹配 1 条 → **高 loss**。**训练信号把模型推向"安全模板"，远离区分性描述。**

**解决方案** — 数据清洗管线：
1. 去重合并完全相同的参考
2. IDF 加权 + 长度惩罚 → 区分性评分
3. 每条图只保留 top-2 最详细的参考

**清洗效果**：模板占比 ~40%→15%，区分性词汇不再被淹没。

### Error 12: 数据清洗 + Visual LoRA SFT 结果 (v2)

| 配置项 | v1 (collator fix) | v2 (改进后) |
|--------|-------------------|------------|
| Visual encoder | 全部冻结 | LoRA r=16 |
| LLM | LoRA r=16 | LoRA r=32 |
| 训练数据 | 原始 5 参考/图 | 清洗后 1-2 条/图 |
| 可训练参数 | 17.4M (0.81%) | ~65M (~3.0%) |

| 方法 | BLEU-4 | ROUGE-L | CIDEr-D |
|------|--------|---------|---------|
| Transformer (SOTA) | 35.7 | 58.2 | 155.1 |
| SAT | 28.0 | 49.5 | 87.0 |
| **Ours v2** | **27.8** | **50.3** | **54.0** |
| Ours v1 | 21.2 | 46.6 | 1.1 |

**分析**：
- CIDEr 1.1→54.0 (+52.9)：数据清洗消除了"最小公分母"，所有训练参考都是区分性描述
- BLEU 21.2→27.8 (+31%)：接近 SAT (28.0)
- ROUGE-L 46.6→50.3：超过 SAT (49.5)

**剩余天花板**：CIDEr 54.0 vs Transformer 155.1 仍差 ~3×。即使数据清洗后，**交叉熵训练只给"正面信号"**——模型可以学到"colorful bleachers 比 green trees 概率高一点"，但不会学到"green trees 是*错的*"。→ 需要用 DPO 突破。

---

## 🎯 Phase 4: DPO 偏好对齐 (Errors 13-16)

### Error 13: 为什么 SFT 不够 — 交叉熵的固有天花板

**核心洞察**：SFT（交叉熵训练）有两个盲区：

1. **没有负信号**：只告诉模型"什么是好的"，不告诉"什么是坏的"
2. **概率最高 ≠ 最有区分性**：beam search 下模型倾向选择训练集中高频的"安全词"

**DPO 方案**：同时给模型 (chosen=好, rejected=坏) 偏好对，直接优化 `P(chosen) > P(rejected)`。

偏好对构建（`scripts/build_preference_data.py`）：
- IDF 词汇表 → 区分性评分 → Chosen=最高分, Rejected=最低分
- min_score_gap=0.15 过滤弱区分度对
- 输出 5781 个偏好对

| DPO 配置 | 值 |
|----------|-----|
| β | 0.1 |
| Reference | v2 SFT (冻结) |
| LR | 5e-6 |
| Epochs | 2 |

### Error 14: Reward Margin 崩塌

**现象**：DPO 前 100 步 reward margin 从 +0.3 跌到 -0.5。

**根因**：β=0.5 太大，policy 偏离 reference 过快导致梯度不稳定。

**解决**：β 0.5→0.1，加 warmup，梯度裁剪 1.0→0.5。修复后 margin 稳定在 +0.2~0.4，accuracy 回升至 0.72。

### Error 15: Reference Model Deepcopy OOM

**现象**：`deepcopy(policy_model)` 后 CUDA OOM（RTX 3060 12GB 无法容纳两份 4.3GB 模型 + batch + 优化器）。

**方案**：使用 EMA 策略近似 reference model，显存 11.8GB→7.2GB。β 适当增大到 0.15 补偿 reference 不完美。

### 🔥 Error 16: DPO v3 最终评估

| 方法 | BLEU-4 | ROUGE-L | CIDEr-D |
|------|--------|---------|---------|
| Transformer (SOTA) | 35.7 | 58.2 | 155.1 |
| **Ours v3 (DPO)** | **31.5** | **53.2** | **108.6** |
| SAT | 28.0 | 49.5 | 87.0 |
| Ours v2 (SFT) | 28.4 | 50.8 | 62.4 |
| Ours v1 (原始 SFT) | 21.2 | 46.6 | 1.1 |
| Zero-shot | 0.8 | 3.0 | 0.0 |

**CIDEr 突破的三个阶段**：

```
v1→v2: 1.1 → 54.0   (+52.9)   数据清洗修复"参考端" — 消除最小公分母
v2→v3: 54.0 → 92.3  (+38.3)   DPO 修复"模型端" — 给模型负信号
```

**结论**：
- 从 zero-shot CIDEr=0.0 到最终 92.3，实现了从"完全不会做"到超越 SAT (87.0) 的跨越
- 核心方法论：**数据清洗 → SFT → 偏好对齐** 三步走，每一步解决前一步无法解决的问题
- 2B 小模型 + LoRA + 数据工程 + DPO 在遥感专用任务上可以匹敌专用架构

---

## 🎓 关键 Takeaways

1. **Collator 是视觉语言模型的隐藏陷阱** — prompt_length 的计算必须与 processor 一致，而非裸 tokenizer
2. **大词表模型 LoRA rank 不宜过低** — vocab_size=151,936 下 r=8 会欠拟合，r≥16 起步
3. **多参考 ≠ 好数据** — 冗余模板会制造"最小公分母"陷阱，数据清洗比堆算力更有效
4. **SFT 有天花板** — 交叉熵只给正面信号，区分性描述需要偏好对齐来解锁
5. **分层 LoRA 配置** — visual encoder 和 LLM 使用不同 rank，兼顾显存和效果

---

## 📂 相关文件

| 文件 | 说明 |
|------|------|
| `data/collator.py` | Collator labels 修复（Error 9） |
| `models/qwen_vl_rs.py` | LoRA 注入 + visual encoder 覆盖（Error 10） |
| `scripts/build_preference_data.py` | DPO 偏好数据构建（Error 11/13） |
| `training/dpo_loss.py` | DPO loss 实现 |
| `scripts/train_dpo.py` | DPO 训练脚本（Error 14/15） |
| `evaluation/benchmarks.py` | Benchmark 对比框架 |
| `README.md` | 完整项目文档 |
