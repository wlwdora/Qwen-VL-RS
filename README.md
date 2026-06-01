# Qwen-VL-RS: Remote Sensing Image Captioning with Vision-Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-orange.svg)](https://github.com/huggingface/peft)
[![DPO](https://img.shields.io/badge/Alignment-DPO-green.svg)](https://arxiv.org/abs/2305.18290)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fine-tuning **Qwen3-VL-2B-Instruct** for remote sensing image captioning via LoRA, data engineering, and DPO preference alignment. Achieves **BLEU-4=29.8, CIDEr-D=92.3** on the RSICD benchmark using a single RTX 3060 12GB GPU.

<p align="center">
  <i>Zero-shot → Collator Fix → Data Cleaning → Visual LoRA → DPO Alignment</i>
  <br>
  <b>CIDEr: 0.0 → 1.1 → 54.0 → 92.3</b>
</p>

---

## 📊 Results

### RSICD Benchmark

| Method | BLEU-4 | ROUGE-L | CIDEr-D | Params |
|--------|--------|---------|---------|--------|
| Transformer (Lu et al., 2019) | 35.7 | 58.2 | 155.1 | — |
| GCN-LSTM | 33.4 | 56.5 | 138.6 | — |
| Up-Down | 31.3 | 54.5 | 124.8 | — |
| Adaptive | 30.1 | 53.6 | 113.9 | — |
| SAT | 28.0 | 49.5 | 87.0 | — |
| **Ours (DPO aligned)** | **29.8** | **51.7** | **92.3** | ~65M (3.0%) |
| Ours (data-cleaned SFT) | 27.8 | 50.3 | 54.0 | ~65M (3.0%) |
| Ours (vanilla SFT) | 21.2 | 46.6 | 1.1 | 17.4M (0.8%) |
| Zero-shot (Qwen3-VL-2B) | 0.8 | 3.0 | 0.0 | — |

> CIDEr-D surpasses SAT (87.0) at 92.3, demonstrating that a 2B VLM with targeted fine-tuning can compete with specialized architectures on domain-specific captioning tasks.

---

## 🗺️ Project Evolution

```
v0 → Zero-shot baseline
     75% empty output, 25% chat-style responses. Base model cannot do image captioning.

v1 → LoRA r=16 + collator bug fix
     Model learns the task format. BLEU-4=21.2, but CIDEr=1.1 — outputs
     are template sentences ("many buildings and green trees...").

v2 → Data cleaning + visual encoder LoRA
     IDF-based deduplication eliminates "least common denominator" effect
     in multi-reference annotations. Visual encoder adaptation improves
     feature extraction. CIDEr 1.1→54.0, BLEU-4 21.2→27.8.

v3 → DPO preference alignment
     5,781 preference pairs (detailed vs. template captions) teach the
     model to prefer distinctive vocabulary. CIDEr 54.0→92.3 (+71%).
     Surpasses SAT baseline.
```

### Key Technical Fixes

1. **Collator labels masking bug** — `_collate_pil()` used bare `tokenizer()` to compute prompt length (19 tokens), but `processor()` expands vision tokens to 270+. Labels only masked the first 19, making 270 `<|image_pad|>` tokens into training targets. **98.5% of compute was wasted predicting padding.** ([error.md#error-9](error.md))

2. **Visual encoder LoRA gap** — Qwen3-VL's ViT uses `qkv/proj/linear_fc1/linear_fc2` layer names, completely different from the LLM's `q_proj/k_proj/v_proj/gate/up/down_proj`. All 96 Linear layers in the 24 visual blocks were frozen. ([error.md#error-10](error.md))

3. **Multi-reference "least common denominator"** — RSICD provides 5 references per image, with 94.2% containing duplicates. Cross-entropy training pushes the model toward generic templates that match multiple references, destroying CIDEr. Solved via IDF-weighted deduplication. ([error.md#error-11](error.md))

> 📖 Full development history with 16 documented errors and solutions: **[error.md](error.md)**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│  Qwen3-VL-2B-Instruct (HuggingFace)         │
│                                              │
│  ┌─────────────────┐    ┌─────────────────┐  │
│  │ Visual Encoder   │    │ Language Model   │  │
│  │ (ViT, 24 blocks) │───▶│ (Qwen3, 28 layers)│  │
│  │                  │    │                  │  │
│  │ LoRA: qkv, proj, │    │ LoRA: q/k/v/o,   │  │
│  │  linear_fc1/2    │    │  gate/up/down    │  │
│  │ r=16             │    │ r=32             │  │
│  └─────────────────┘    └─────────────────┘  │
│                                              │
│  Trainable: ~65M / 2.1B (3.0%)               │
└─────────────────────────────────────────────┘
```

| Component | Module | Target Layers | LoRA Rank |
|-----------|--------|--------------|-----------|
| Visual Encoder | 24× ViT blocks | `qkv`, `proj`, `linear_fc1`, `linear_fc2` | r=16 |
| Language Model | 28× decoder layers | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | r=32 |

---

## 📁 Project Structure

```
Qwen-VL-RS/
├── models/
│   └── qwen_vl_rs.py              # Qwen3-VL + LoRA wrapper (load/inject/generate/merge)
├── data/
│   ├── dataset.py                 # RSICD/UCM/Sydney Dataset
│   ├── transforms.py              # Remote sensing augmentations
│   ├── collator.py                # Multimodal data collator
│   └── prompts.py                 # Instruction templates
├── training/
│   ├── trainer.py                 # Training loop (OneCycleLR, early stop, grad accum)
│   ├── loss.py                    # Cross-entropy & focal loss
│   ├── metrics.py                 # BLEU/ROUGE/CIDEr/METEOR/SPICE/CHAIR
│   └── dpo_loss.py                # DPO loss with reference model
├── evaluation/
│   ├── eval.py                    # Evaluation engine (generation + metrics)
│   ├── benchmarks.py              # Multi-method comparison & report generation
│   └── error_analysis.py          # CHAIR hallucination, land cover F1
├── inference/
│   ├── infer.py                   # Single/batch inference
│   ├── gradio_app.py              # Interactive web demo
│   └── vllm_serve.py              # High-throughput serving
├── scripts/
│   ├── train_rsicd_r16.py         # LoRA fine-tuning script
│   ├── train_dpo.py               # DPO alignment training
│   ├── build_preference_data.py   # Preference pair construction
│   ├── eval_rsicd.py              # RSICD evaluation
│   ├── eval_zero_shot.py          # Zero-shot baseline
│   ├── run_benchmark_full.py      # Benchmark report generation
│   └── preprocess_data.py         # Data preprocessing
├── configs/                       # YAML configuration files
├── experiments/
│   ├── benchmarks/                # Benchmark reports (Markdown + LaTeX)
│   └── evaluations/               # Evaluation result JSONs
├── error.md                       # Development log: 16 errors & solutions
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/wlwdora/Qwen-VL-RS.git
cd Qwen-VL-RS
pip install -r requirements.txt
```

### Requirements

- Python 3.10+, PyTorch 2.7+, CUDA 11.8+
- GPU with ≥12GB VRAM (tested on RTX 3060)
- Qwen3-VL-2B-Instruct (local path or HuggingFace)

### Data Preparation

```bash
# Download RSICD manually from GitHub (see error.md #4 for details)
# Place under data/raw/RSICD_optimal/

# Preprocess
python scripts/preprocess_data.py
```

### Training

```bash
# Stage 1: LoRA fine-tuning
python scripts/train_rsicd_r16.py

# Stage 2: Build preference pairs
python scripts/build_preference_data.py

# Stage 3: DPO alignment
python scripts/train_dpo.py
```

### Evaluation

```bash
# Evaluate trained model
python scripts/eval_rsicd.py

# Zero-shot baseline
python scripts/eval_zero_shot.py

# Generate benchmark report
python scripts/run_benchmark_full.py
```

### Inference

```python
from models.qwen_vl_rs import QwenVLForRemoteSensing
from PIL import Image

model = QwenVLForRemoteSensing.from_pretrained(
    model_path="Qwen/Qwen3-VL-2B-Instruct",
    lora_adapter_path="output/qwen_vl_rs_dpo/best_model",
)
image = Image.open("scene.jpg").convert("RGB")
caption = model.predict(image, prompt="Describe this remote sensing image in detail.")
```

---

## 📐 Metrics

| Metric | Description | Best For |
|--------|-------------|----------|
| **CIDEr-D** | TF-IDF weighted n-gram consensus | Caption distinctiveness (primary) |
| **BLEU-4** | 4-gram precision with brevity penalty | Surface-level word matching |
| **ROUGE-L** | Longest common subsequence recall | Keyword coverage |
| METEOR | Synonym + stem matching (requires Java) | Semantic similarity |
| SPICE | Scene graph proposition evaluation | Spatial/logical correctness |
| CHAIR | Object hallucination rate | Factual accuracy |

> **Note**: METEOR and SPICE require Java runtime and are unavailable on this Windows environment. BLEU/ROUGE/CIDEr provide sufficient coverage.

---

## 🔬 Ablation Study

| Change | BLEU-4 Δ | CIDEr-D Δ | Key Insight |
|--------|----------|-----------|-------------|
| Collator bug fix | +20.4 | +1.1 | 98.5% compute was wasted on padding |
| Data deduplication | +6.6 | +52.9 | Eliminates "least common denominator" in multi-ref training |
| Visual encoder LoRA | embedded | embedded | ViT adaptation critical for overhead imagery |
| DPO alignment | +2.0 | +38.3 | Preference signals break through cross-entropy ceiling |
| Prompt language (CN→EN) | 0.0 | 0.0 | Cross-lingual mismatch is not a bottleneck |

---

## 📝 Development Log

`error.md` documents 16 errors encountered during development, organized into four phases:

| Phase | Errors | Highlights |
|-------|--------|------------|
| 🔧 Environment & Pipeline | #1–6 | CUDA segfault, albumentations API, chat template |
| 🏋️ Training Debugging | #7–9 | LoRA underfitting, **collator labels bug** |
| 📊 Data & Architecture | #10–12 | **Visual encoder gap**, **CIDEr "LCD" effect** |
| 🎯 DPO Alignment | #13–16 | SFT ceiling, reward margin collapse, final results |

Each entry includes symptoms, root cause analysis, solution, and code diff. Designed as both a reference and a demonstration of systematic debugging methodology.

---

## 📚 References

| Paper | Focus |
|-------|-------|
| Lu et al., *RSICD: Remote Sensing Image Captioning Dataset* (2018) | Dataset |
| Lu et al., *Exploring Models and Data for Remote Sensing Image Captioning* (2019) | Baselines |
| Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) | Method |
| Rafailov et al., *Direct Preference Optimization* (NeurIPS 2023) | Alignment |
| Bai et al., *Qwen-VL: A Versatile Vision-Language Model* (2023) | Base Model |

---

## 👤 Author

**wlwdora** — School of Electronic Information, Wuhan University

- **Research**: Multimodal vision-language models, remote sensing, efficient fine-tuning, preference alignment
- **Contact**: 2021302121165@whu.edu.cn

---

## 📄 License

MIT License — academic research and educational use.
