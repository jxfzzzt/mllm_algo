"""LLaVA 的视觉投影层: 把视觉特征映射到 LLM 的词向量空间。

    linear      LLaVA-1.0, 一层线性层
    mlp2x_gelu  LLaVA-1.5, 两层 MLP, 效果明显好于线性层
"""
import torch.nn as nn


def build_projector(projector_type: str, vision_dim: int, llm_dim: int) -> nn.Module:
    if projector_type == "linear":
        return nn.Linear(vision_dim, llm_dim)
    if projector_type == "mlp2x_gelu":
        return nn.Sequential(nn.Linear(vision_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim))
    raise ValueError(f"未知的 projector 类型: {projector_type}")
