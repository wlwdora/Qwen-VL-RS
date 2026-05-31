# 遥感图像描述 —— 推理模块
# 单图/批量推理、Gradio 交互 Demo、vLLM 高性能部署

from .infer import InferenceEngine
from .gradio_app import launch_demo
from .vllm_serve import CustomInferenceEngine, create_app

# 向后兼容别名
VLLMServer = CustomInferenceEngine

__all__ = [
    "InferenceEngine",
    "launch_demo",
    "VLLMServer",
    "CustomInferenceEngine",
    "create_app",
]
