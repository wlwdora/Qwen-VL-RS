# Qwen-VL-RS Benchmark 结果

> 生成日期: 2026-06-01 18:39:03

## RSICD

| 方法 | BLEU-4 | METEOR | ROUGE-L | CIDER-D | SPICE |
|---|---|---|---|---|---|
| Transformer | 35.7 | 31.5 | 58.2 | 155.1 | 36.5 |
| GCN-LSTM | 33.4 | 30.0 | 56.5 | 138.6 | 33.2 |
| Up-Down | 31.3 | 29.1 | 54.5 | 124.8 | 31.1 |
| Adaptive | 30.1 | 28.3 | 53.6 | 113.9 | 29.0 |
| Ours v3 (DPO 偏好对齐) **(ours)** | 29.8 | 0.0 | 51.7 | 92.3 | 0.0 |
| SAT | 28.0 | 26.8 | 49.5 | 87.0 | 23.5 |
| Ours v2 (数据清洗+visual LoRA SFT) **(ours)** | 27.8 | 0.0 | 50.3 | 54.0 | 0.0 |
| Ours v1 (collator fix, 原始数据 SFT) **(ours)** | 21.2 | 0.0 | 46.6 | 1.1 | 0.0 |
| Zero-shot (Qwen3-VL-2B) | 0.8 | 0.0 | 3.0 | 0.0 | 0.0 |
