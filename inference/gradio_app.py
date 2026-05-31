"""
Gradio 交互式遥感图像描述 Demo。

功能：
  - 图像上传（支持 jpg / png / tif 格式）
  - 并排对比：我们的模型 vs GPT-4V vs BLIP-2
  - 可调节的生成参数（温度、最大 token 数）
  - 一键导出结果为 JSON
  - 精选遥感图像示例库
"""

import gradio as gr
import logging

logger = logging.getLogger(__name__)


def launch_demo(checkpoint_path: str = None, share: bool = False):
    """启动 Gradio Web 界面。

    参数：
        checkpoint_path: 微调模型 checkpoint 路径。
        share: 是否创建公开分享链接。
    """
    # TODO: 实现 Gradio 交互 Demo
    # 1. 加载模型
    # 2. 构建 UI 布局：
    #    - 左侧：图像上传 + 参数滑条
    #    - 右侧：模型输出 + 多模型对比面板
    # 3. 绑定回调函数
    # 4. 启动服务
    raise NotImplementedError("【待实现】Gradio Demo")


if __name__ == "__main__":
    launch_demo()
