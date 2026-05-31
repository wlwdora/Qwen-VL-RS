"""
Gradio 交互式遥感图像描述 Demo。

功能：
  - 图像上传（支持 jpg / png / tif 格式）
  - 可调节的生成参数（温度、最大 token 数）
  - 支持多 prompt 对比模式
  - 一键导出结果为 JSON
  - 精选遥感图像示例库

用法：
    python inference/gradio_app.py --checkpoint output/qwen_vl_rs_lora/best_model

环境变量：
    QWEN_MODEL_PATH: 基座模型路径（默认 D:/Qwen/Qwen3-VL-2B-Instruct）
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class GradioRemoteSensingDemo:
    """Gradio 交互 Demo 的完整封装。"""

    def __init__(
        self,
        checkpoint_path: str,
        base_model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        参数：
            checkpoint_path: 训练好的 LoRA adapter 路径。
            base_model_path: 基座模型路径。
            device: "cuda" 或 "cpu"。
        """
        self.checkpoint_path = checkpoint_path
        self.base_model_path = base_model_path or os.environ.get(
            "QWEN_MODEL_PATH", "D:/Qwen/Qwen3-VL-2B-Instruct"
        )
        self.device = device
        self.model_wrapper = None

        # ── 遥感示例图像（可替换为实际路径）───
        self.example_dir = Path("data/examples")

        # ── 预设 prompt ──────────────────────
        self.presets = {
            "详细描述": "请详细描述这张遥感图像的内容，包括地物类型、空间分布和显著特征。",
            "简洁描述": "用一句话描述这张遥感图像。",
            "土地覆盖": "请识别这张遥感图像中的主要土地覆盖类型。",
            "变化检测": "请比较这张遥感图像中的变化区域，描述有什么不同。",
            "灾害评估": "请评估这张遥感图像中可能存在的自然灾害影响。",
            "城市规划": "请从城市规划角度描述这张遥感图像中的功能区分布。",
        }

    def load_model(self):
        """加载模型和处理器。"""
        logger.info(f"正在加载模型: base={self.base_model_path}, adapter={self.checkpoint_path}")

        import torch
        from models.qwen_vl_rs import QwenVLForRemoteSensing
        self._torch = torch

        self.model_wrapper = QwenVLForRemoteSensing.from_pretrained(
            model_path=self.base_model_path,
            lora_adapter_path=self.checkpoint_path
            if os.path.isdir(self.checkpoint_path)
            else None,
            torch_dtype=torch.float16,
        )
        logger.info("模型加载完成")

    def generate_caption(
        self,
        image: np.ndarray,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[str, float]:
        """为单张图像生成描述。

        返回：
            (caption, elapsed_time_seconds)
        """
        if self.model_wrapper is None:
            self.load_model()

        # numpy array → PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.model_wrapper.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.model_wrapper.processor(
            text=[text], images=[pil_image], return_tensors="pt"
        ).to(self.model_wrapper.peft_model.device)

        # 生成
        import torch
        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model_wrapper.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0),
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, prompt_len:]
        caption = self.model_wrapper.decode(new_tokens, skip_special_tokens=True)
        elapsed = time.time() - t0

        return caption.strip(), round(elapsed, 2)

    def compare_prompts(
        self,
        image: np.ndarray,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """用多个预设 prompt 生成描述并返回对比表格。"""
        lines = ["| 模式 | 描述 |", "|:---|:---|"]
        for preset_name, prompt in self.presets.items():
            caption, elapsed = self.generate_caption(
                image, prompt, max_new_tokens, temperature
            )
            lines.append(f"| **{preset_name}** | {caption} ({elapsed:.1f}s) |")
        return "\n".join(lines)

    def export_result(
        self,
        image: np.ndarray,
        caption: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """导出当前结果到 JSON 文件。"""
        export_dir = Path("experiments/gradio_exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = export_dir / f"result_{timestamp}.json"

        result = {
            "timestamp": timestamp,
            "prompt": prompt,
            "caption": caption,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return str(export_path)

    def get_example_images(self) -> List[str]:
        """获取示例图像列表。"""
        if not self.example_dir.exists():
            return []
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        examples = [
            str(p) for p in self.example_dir.iterdir()
            if p.suffix.lower() in exts
        ]
        return sorted(examples)[:10]


# ════════════════════════════════════════════════════════════════
# UI 构建
# ════════════════════════════════════════════════════════════════

def launch_demo(
    checkpoint_path: str = None,
    base_model_path: str = None,
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
):
    """启动 Gradio Web 界面。

    参数：
        checkpoint_path: 微调模型 checkpoint 路径。
        base_model_path: 基座模型路径。为 None 时从环境变量读取。
        share: 是否创建公开分享链接。
        server_name: 服务器地址（0.0.0.0 允许外部访问）。
        server_port: 服务器端口。
    """
    import gradio as gr

    if checkpoint_path is None:
        checkpoint_path = os.environ.get(
            "QWEN_ADAPTER_PATH",
            "output/qwen_vl_rs_lora/best_model",
        )

    demo = GradioRemoteSensingDemo(
        checkpoint_path=checkpoint_path,
        base_model_path=base_model_path,
    )

    # ── 加载模型（启动时预热）──────────────
    try:
        demo.load_model()
        model_status = "✅ 模型已加载"
    except Exception as e:
        model_status = f"⚠️ 模型加载失败: {e}"

    # ── 示例图像 ───────────────────────────
    example_images = demo.get_example_images()

    # ── UI 布局 ────────────────────────────
    with gr.Blocks(
        title="Qwen-VL-RS — 遥感图像描述",
        theme=gr.themes.Soft(),
        css="""
        .output-text textarea { font-size: 16px !important; }
        .status-box { padding: 8px; border-radius: 8px; margin: 8px 0; }
        """,
    ) as app:
        gr.Markdown(
            """
            # 🛰️ Qwen-VL-RS — 遥感图像描述系统
            基于 **Qwen3-VL-2B-Instruct + LoRA** 微调的遥感图像智能描述模型。
            上传遥感图像即可自动生成中文描述。
            """
        )

        # ── 状态栏 ────────────────────────
        gr.Markdown(f"**状态**: {model_status}")

        with gr.Row():
            # ── 左侧：控制面板 ──────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📷 输入")

                input_image = gr.Image(
                    label="上传遥感图像",
                    type="numpy",
                    image_mode="RGB",
                    height=300,
                )

                # 示例
                if example_images:
                    gr.Examples(
                        examples=example_images,
                        inputs=input_image,
                        label="示例图像",
                    )

                prompt_dropdown = gr.Dropdown(
                    choices=list(demo.presets.keys()),
                    value="详细描述",
                    label="描述模式",
                )

                custom_prompt = gr.Textbox(
                    value=demo.presets["详细描述"],
                    label="Prompt（可自定义）",
                    lines=2,
                )

                # 同步 dropdown 和 textbox
                def update_prompt(preset_name):
                    return demo.presets.get(preset_name, "")

                prompt_dropdown.change(
                    fn=update_prompt,
                    inputs=prompt_dropdown,
                    outputs=custom_prompt,
                )

                gr.Markdown("### ⚙️ 参数")

                max_tokens = gr.Slider(
                    minimum=16, maximum=512, value=256, step=16,
                    label="最大生成长度",
                )
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.7, step=0.05,
                    label="温度（越高越随机）",
                )
                top_p = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.9, step=0.05,
                    label="Top-p",
                )

                generate_btn = gr.Button(
                    "🚀 生成描述", variant="primary", size="lg"
                )

            # ── 右侧：输出面板 ──────────────
            with gr.Column(scale=2):
                gr.Markdown("### 📝 输出")

                output_text = gr.Textbox(
                    label="生成描述",
                    lines=5,
                    elem_classes=["output-text"],
                )

                output_time = gr.Textbox(
                    label="推理耗时", visible=True,
                )

                with gr.Accordion("🔬 多 Prompt 对比", open=False):
                    compare_btn = gr.Button("运行对比")
                    compare_output = gr.Markdown(
                        "点击「运行对比」查看多个 prompt 的生成结果"
                    )

                with gr.Accordion("💾 导出", open=False):
                    export_btn = gr.Button("导出为 JSON")
                    export_path = gr.Textbox(
                        label="导出路径", interactive=False
                    )

                with gr.Accordion("📊 输入信息", open=False):
                    input_info = gr.JSON(label="参数摘要")

        # ── 回调绑定 ───────────────────────

        def on_generate(image, prompt, max_t, temp, top_p_val):
            if image is None:
                return "⚠️ 请先上传图像", "", {}
            try:
                caption, elapsed = demo.generate_caption(
                    image, prompt, max_t, temp, top_p_val,
                )
                info = {
                    "prompt": prompt,
                    "max_new_tokens": max_t,
                    "temperature": temp,
                    "top_p": top_p_val,
                    "elapsed_seconds": elapsed,
                }
                return caption, f"⏱️ {elapsed:.2f}s", info
            except Exception as ex:
                return f"❌ 错误: {ex}", "", {}

        generate_btn.click(
            fn=on_generate,
            inputs=[input_image, custom_prompt, max_tokens, temperature, top_p],
            outputs=[output_text, output_time, input_info],
        )

        def on_compare(image, max_t, temp):
            if image is None:
                return "⚠️ 请先上传图像"
            return demo.compare_prompts(image, max_t, temp)

        compare_btn.click(
            fn=on_compare,
            inputs=[input_image, max_tokens, temperature],
            outputs=compare_output,
        )

        def on_export(image, caption, prompt, max_t, temp):
            path = demo.export_result(image, caption, prompt, max_t, temp)
            return path

        export_btn.click(
            fn=on_export,
            inputs=[input_image, output_text, custom_prompt, max_tokens, temperature],
            outputs=export_path,
        )

    # ── 启动 ─────────────────────────────
    logger.info(f"启动 Gradio 服务于 {server_name}:{server_port}")
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen-VL-RS Gradio Demo")
    parser.add_argument("--checkpoint", type=str,
                        default="output/qwen_vl_rs_lora/best_model",
                        help="LoRA adapter checkpoint 路径")
    parser.add_argument("--base_model", type=str, default=None,
                        help="基座模型路径")
    parser.add_argument("--share", action="store_true",
                        help="创建公开分享链接")
    parser.add_argument("--port", type=int, default=7860,
                        help="服务器端口")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="服务器地址")

    args = parser.parse_args()

    import torch
    launch_demo(
        checkpoint_path=args.checkpoint,
        base_model_path=args.base_model,
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )
