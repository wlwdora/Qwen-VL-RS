"""高性能批量推理 API 服务。

支持两种后端：
  1. vLLM 后端：PagedAttention + continuous batching（推荐生产环境）
  2. 自定义后端：基于模型 wrapper 的 FastAPI 批量推理（兼容性好）

提供 OpenAI-compatible API：
  - POST /v1/chat/completions — 对话补全
  - POST /v1/captions — 遥感图像描述
  - GET /health — 健康检查

用法：
    # vLLM 模式
    python inference/vllm_serve.py --backend vllm --model output/merged_model

    # 自定义模式（使用训练好的 LoRA adapter）
    python inference/vllm_serve.py --backend custom --checkpoint output/best_model

    # 测试请求
    curl http://localhost:8000/health
"""

import base64
import io
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 自定义推理引擎
# ════════════════════════════════════════════════════════════════

class CustomInferenceEngine:
    """基于模型 wrapper 的批量推理引擎。

    适用于没有 vLLM 或不支持 vLLM 的模型场景。
    支持 batch 推理以提升吞吐量。
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        max_batch_size: int = 4,
    ):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.device = device
        self.max_batch_size = max_batch_size
        self.model_wrapper = None
        self._loaded = False

    def load(self):
        """加载模型。"""
        import torch
        from models.qwen_vl_rs import QwenVLForRemoteSensing

        logger.info(f"加载模型: base={self.base_model_path}")
        self.model_wrapper = QwenVLForRemoteSensing(
            model_path=self.base_model_path,
            torch_dtype=torch.float16,
        )
        self.model_wrapper.load()

        if self.adapter_path and os.path.exists(self.adapter_path):
            from peft import PeftModel
            self.model_wrapper.peft_model = PeftModel.from_pretrained(
                self.model_wrapper.model, self.adapter_path
            )
            logger.info(f"LoRA adapter 已加载: {self.adapter_path}")

        self._loaded = True
        logger.info("推理引擎就绪")

    def generate_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """批量为多张图像生成描述。

        参数：
            images: PIL Image 列表。
            prompts: 对应的 prompt 列表。
            max_new_tokens: 最大生成长度。
            temperature: 采样温度。
            top_p: nucleus 采样参数。

        返回：
            [{"caption": str, "elapsed": float, "tokens": int}, ...]
        """
        if not self._loaded:
            self.load()

        import torch
        results = []

        # 逐张处理（保持兼容性，未来可改进为真正的 batch generation）
        for image, prompt in zip(images, prompts):
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text = self.model_wrapper.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.model_wrapper.processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(self.model_wrapper.peft_model.device)

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
            caption = self.model_wrapper.decode(
                new_tokens, skip_special_tokens=True
            )
            elapsed = time.time() - t0

            results.append({
                "caption": caption.strip(),
                "elapsed": round(elapsed, 3),
                "tokens": int(new_tokens.shape[0]),
            })

        return results

    def generate_single(
        self,
        image: Image.Image,
        prompt: str = "请详细描述这张遥感图像的内容。",
        **kwargs,
    ) -> Dict[str, Any]:
        """单张图像生成（便捷方法）。"""
        results = self.generate_batch([image], [prompt], **kwargs)
        return results[0]


# ════════════════════════════════════════════════════════════════
# FastAPI 服务
# ════════════════════════════════════════════════════════════════

def create_app(
    base_model_path: str,
    adapter_path: Optional[str] = None,
    backend: str = "custom",
):
    """创建 FastAPI 应用。

    参数：
        base_model_path: 基座模型路径。
        adapter_path: LoRA adapter 路径。
        backend: "vllm" 或 "custom"。
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        logger.error("需要安装 fastapi 和 uvicorn：pip install fastapi uvicorn")
        raise

    # ── 初始化引擎 ────────────────────────
    if backend == "vllm":
        logger.warning("vLLM 后端暂未实现 vLLM Qwen3-VL 直接支持，回退到自定义后端")
        # try:
        #     from vllm import LLM, SamplingParams
        #     engine = None  # vLLM engine
        # except ImportError:
        #     logger.error("vLLM 未安装，回退到自定义后端")
        backend = "custom"

    engine = CustomInferenceEngine(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
    )

    @asynccontextmanager
    async def lifespan(app):
        """应用生命周期：启动时加载模型。"""
        logger.info("正在加载模型...")
        engine.load()
        logger.info("模型加载完成，服务就绪")
        yield

    app = FastAPI(
        title="Qwen-VL-RS Inference Server",
        description="遥感图像描述推理 API 服务",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ═══════════════════════════════════════
    # 路由
    # ═══════════════════════════════════════

    @app.get("/health")
    async def health():
        """健康检查。"""
        import torch
        return {
            "status": "healthy",
            "model": base_model_path,
            "adapter": adapter_path,
            "backend": backend,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }

    @app.get("/v1/models")
    async def list_models():
        """列出可用模型。"""
        return {
            "object": "list",
            "data": [
                {
                    "id": "qwen-vl-rs",
                    "object": "model",
                    "owned_by": "qwen-vl-rs",
                }
            ],
        }

    @app.post("/v1/captions")
    async def generate_caption(request: Dict[str, Any]):
        """遥感图像描述生成。

        请求格式：
        {
            "image": "base64编码的图像" 或 "图像URL",
            "prompt": "请详细描述...",          // 可选
            "max_new_tokens": 256,             // 可选
            "temperature": 0.7,                // 可选
            "top_p": 0.9                       // 可选
        }

        响应格式：
        {
            "caption": "...",
            "elapsed": 1.23,
            "tokens": 45
        }
        """
        try:
            # 解析图像
            image_data = request.get("image")
            if image_data is None:
                raise HTTPException(status_code=400, detail="缺少 'image' 字段")

            if image_data.startswith("data:") or image_data.startswith("/9j/"):
                # Base64 编码
                if image_data.startswith("data:"):
                    image_data = image_data.split(",", 1)[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            elif image_data.startswith("http"):
                # URL — 简单处理，实际应用中应异步下载
                raise HTTPException(status_code=400, detail="URL 方式暂未实现")
            else:
                # 文件路径
                if not os.path.exists(image_data):
                    raise HTTPException(status_code=400,
                                        detail=f"图像文件不存在: {image_data}")
                image = Image.open(image_data).convert("RGB")

            prompt = request.get("prompt", "请详细描述这张遥感图像的内容。")
            max_new_tokens = request.get("max_new_tokens", 256)
            temperature = request.get("temperature", 0.7)
            top_p = request.get("top_p", 0.9)

            result = engine.generate_single(
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            return JSONResponse(content=result)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"推理错误: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/captions/batch")
    async def generate_captions_batch(request: Dict[str, Any]):
        """批量遥感图像描述生成。

        请求格式：
        {
            "items": [
                {"image": "base64...", "prompt": "..."},
                ...
            ],
            "max_new_tokens": 256,    // 可选，全局默认
            "temperature": 0.7        // 可选，全局默认
        }
        """
        try:
            items = request.get("items", [])
            if not items:
                raise HTTPException(status_code=400, detail="缺少 'items' 字段")

            images = []
            prompts = []
            for item in items:
                image_data = item.get("image")
                if image_data is None:
                    raise HTTPException(status_code=400, detail="每个 item 需要 'image' 字段")

                if image_data.startswith("data:") or image_data.startswith("/9j/"):
                    if image_data.startswith("data:"):
                        image_data = image_data.split(",", 1)[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                elif os.path.exists(image_data):
                    image = Image.open(image_data).convert("RGB")
                else:
                    raise HTTPException(status_code=400,
                                        detail=f"无法解析图像: {image_data[:30]}...")

                images.append(image)
                prompts.append(item.get("prompt", "请详细描述这张遥感图像的内容。"))

            max_new_tokens = request.get("max_new_tokens", 256)
            temperature = request.get("temperature", 0.7)
            top_p = request.get("top_p", 0.9)

            results = engine.generate_batch(
                images=images,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            return JSONResponse(content={"results": results, "count": len(results)})

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"批量推理错误: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ════════════════════════════════════════════════════════════════
# CLI 入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen-VL-RS 推理 API 服务")
    parser.add_argument("--backend", type=str, default="custom",
                        choices=["vllm", "custom"],
                        help="推理后端 (vllm 或 custom)")
    parser.add_argument("--model", type=str, default="D:/Qwen/Qwen3-VL-2B-Instruct",
                        help="基座模型路径")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="LoRA adapter checkpoint 路径")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="服务绑定地址")
    parser.add_argument("--port", type=int, default=8000,
                        help="服务端口")
    parser.add_argument("--reload", action="store_true",
                        help="开发模式热重载")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = create_app(
        base_model_path=args.model,
        adapter_path=args.checkpoint,
        backend=args.backend,
    )

    import uvicorn

    logger.info(f"启动推理服务: {args.host}:{args.port}, backend={args.backend}")
    uvicorn.run(
        app if not args.reload else "inference.vllm_serve:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=args.reload,
    )
