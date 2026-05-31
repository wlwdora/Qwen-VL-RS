# CLAUDE.md — Qwen-VL-RS 项目上下文

此文件由 Claude 在 2026-05-31 创建，用于在新会话中恢复项目上下文。

## 项目定位

基于 Qwen3-VL-2B-Instruct + LoRA 的遥感图像描述（Remote Sensing Image Captioning）。
目标：微调后的小模型在遥感描述任务上超越 GPT-4V zero-shot 效果。
用途：武大电子信息研0学生找日常实习的项目。

## 关键路径

- 项目根目录：`D:\work\Qwen-VL-RS`
- 基座模型：`D:\Qwen\Qwen3-VL-2B-Instruct`（Qwen3-VL 2B 参数，本地缓存）
- 旧项目（参考）：`D:\Qwen`（手势识别 SFT/DPO/VLA，不再开发）

## 项目结构速查

```
configs/          → 训练/数据/LoRA 配置（YAML）
data/             → 数据集加载、增强、collator、prompt 模板
models/           → Qwen3-VL + LoRA 封装
training/         → 手写 Trainer、loss、metrics（非 MS-SWIFT）
evaluation/       → 评估引擎、benchmark、错误分析
inference/        → 推理引擎、Gradio Demo、vLLM 服务
scripts/          → train_lora.sh, eval.sh 等
experiments/      → 实验记录模板 + 日志
notebooks/        → 规划 4 个 Jupyter notebook
tests/            → 单元测试
```

## 当前状态（2026-05-31）

**已完成：**
- 完整的项目框架和目录结构
- README.md（中文，含实验设计、面试要点）
- 所有模块的 `__init__.py` 和占位文件（docstring + TODO）
- Qwen-VL-RS-项目完全指南.docx（57KB Word文档）
- 3 个 YAML 配置文件
- 4 个 bash 脚本
- .gitignore, requirements.txt
- 实验记录模板 exp_template.md

**待实现（所有 .py 文件仅有骨架）：**
- data/dataset.py → RSICD/UCM/Sydney JSONL 解析
- data/transforms.py → 遥感图像增强管线
- data/collator.py → 多模态批次整理
- models/qwen_vl_rs.py → 模型加载 + LoRA 注入
- training/trainer.py → HF Trainer 训练循环
- training/loss.py → 损失函数
- training/metrics.py → CIDEr/BLEU/METEOR/ROUGE/SPICE
- evaluation/ → 评估和错误分析
- inference/ → 推理和 Gradio Demo
- tests/ → 单元测试

**下一步：从 `data/dataset.py` 开始实现，先下载 RSICD 数据集。**

## 重要设计决策

1. **不用 MS-SWIFT**：手写 HuggingFace Trainer，面试时可解释每一行代码
2. **LoRA rank=16**：默认值，消融实验测试 {8,16,32,64}
3. **CIDEr-D 作为首要指标**：TF-IDF 加权，比 BLEU 更适合图像描述评估
4. **7 组消融实验**：LoRA rank、target modules、数据量、prompt、增强、多数据集、解冻策略
5. **4 个 baseline**：Qwen-VL zero-shot、GPT-4V zero-shot、BLIP-2 LoRA、Ours
6. **遥感专用增强**：离散旋转（0/90/180/270）、多尺度裁剪、光谱扰动
7. **全中文注释**：用户偏好，所有文档和代码注释使用中文

## 用户偏好

- 初学者水平，需要名词解释和设计决策的"为什么"
- 用户说"中文"时指所有文档注释用中文
- D 盘写入需要特殊处理（根目录无权限，用 D:\work\）
- 武大电子信息背景，遥感是武大王牌学科
