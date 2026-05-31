# 实验：[编号] —— [简短描述]

**日期**：YYYY-MM-DD
**作者**：[你的名字]
**状态**：[规划中 | 运行中 | 已完成 | 已废弃]

---

## 1. 实验假设

<!-- 你预期会发生什么？为什么？ -->

## 2. 实验配置

```yaml
# 与 baseline 不同的关键参数
model: ...
lora_rank: ...
learning_rate: ...
dataset: ...
```

## 3. 实验结果

| 指标 | Baseline | 本次实验 | Δ |
|:---|---:|---:|---:|
| BLEU-4 | | | |
| METEOR | | | |
| CIDEr-D | | | |
| ROUGE-L | | | |
| SPICE | | | |
| Train Loss | | | |
| Eval Loss | | | |

## 4. 训练过程观察

<!-- loss 曲线是否正常？有没有出现 loss spike 或 plateau？训练速度是否符合预期？ -->

## 5. 错误分析

<!-- 模型在哪些类型的样本上仍然失败？举 3-5 个具体例子 -->

## 6. 结论与下一步

<!-- 假设是否成立？学到了什么？下一步做什么？ -->

## 7. 产物清单

- 模型 checkpoint：`output/...`
- TensorBoard 日志：`experiments/logs/...`
- W&B 运行记录：`https://wandb.ai/...`
