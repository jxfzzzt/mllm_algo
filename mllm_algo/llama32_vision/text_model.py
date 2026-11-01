"""Llama 3.2-Vision 的语言模型: Llama 3.1 + 每隔 4 层 self-attention 插入 1 层门控 cross-attention。

以 11B 为例: 一共 40 层, 第 [3, 8, 13, 18, 23, 28, 33, 38] 层是 cross-attention 层, 其余 32 层就是 Llama 3.1-8B。
训练时 Llama 3.1 的参数全部冻结, 只训练 cross-attention 层; 没有图像输入时跳过这些层,
模型退化成原始的 Llama 3.1, 因此纯文本能力完全不受影响 (可以直接替换 Llama 3.1)。

Llama 的标准组件: RMSNorm、RoPE、GQA (分组查询注意力)、SwiGLU。
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import causal_padding_mask


class RMSNorm(nn.Module):
    """x / RMS(x) * weight, 相比 LayerNorm 去掉了减均值和 bias。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        x_fp32 = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x_fp32.type_as(x)


# ---------------------------------------------------------------------
# RoPE 旋转位置编码 (Llama 3.1 还有一个长度外推的频率缩放, 这里省略)
# ---------------------------------------------------------------------
def rope_cos_sin(position_ids: torch.Tensor, head_dim: int, theta: float = 500000.0):
    """position_ids: [B, L] -> cos / sin: [B, 1, L, head_dim]"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=position_ids.device).float() / head_dim))
    freqs = position_ids[..., None].float() * inv_freq  # [B, L, head_dim / 2]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[:, None], emb.sin()[:, None]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: 每个 KV head 被 n_rep 个 query head 共享。[B, H_kv, L, d] -> [B, H_kv * n_rep, L, d]"""
    return x.repeat_interleave(n_rep, dim=1)


# ---------------------------------------------------------------------
# Attention & MLP
# ---------------------------------------------------------------------
class SelfAttention(nn.Module):
    """GQA + RoPE 的因果自注意力。"""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(self, x, cos, sin, mask):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        n_rep = self.num_heads // self.num_kv_heads
        out = F.scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), attn_mask=mask[:, None])
        return self.o_proj(out.transpose(1, 2).reshape(B, L, -1))


class CrossAttention(nn.Module):
    """文本 query 对视觉 token 做 cross-attention。

    和 self-attention 的区别: K/V 来自视觉特征; 不加 RoPE (视觉 token 没有文本位置);
    q、k 额外做 RMSNorm (QK-Norm), 稳定训练。视觉特征是固定的, 推理时 K/V 只需要算一次并缓存。
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, vision_states, mask):
        # x: [B, L, D], vision_states: [B, S, D] (已投影到文本维度), mask: [B, L, S]
        B, L, _ = x.shape
        S = vision_states.shape[1]
        q = self.q_norm(self.q_proj(x).view(B, L, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(vision_states).view(B, S, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(vision_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        n_rep = self.num_heads // self.num_kv_heads
        out = F.scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), attn_mask=mask[:, None])
        return self.o_proj(out.transpose(1, 2).reshape(B, L, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------
# Decoder 层
# ---------------------------------------------------------------------
class SelfAttentionDecoderLayer(nn.Module):
    """标准的 Llama 层 (Pre-Norm)。"""

    def __init__(self, dim, num_heads, num_kv_heads, ffn_dim):
        super().__init__()
        self.input_norm = RMSNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, num_kv_heads)
        self.post_attn_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, ffn_dim)

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_norm(x), cos, sin, mask)
        return x + self.mlp(self.post_attn_norm(x))


class CrossAttentionDecoderLayer(nn.Module):
    """门控 cross-attention 层 (和 Flamingo 的 GATED XATTN-DENSE 相同):

        x = x + tanh(gate_attn) · CrossAttn(x, 视觉)
        x = x + tanh(gate_mlp)  · MLP(x)

    gate 初始化为 0, 训练开始时这一层是恒等映射, 不会破坏冻结的 Llama。
    text_row_mask 标记哪些文本 token 前面有图像, 没有图像的 token 不接收视觉信息。
    """

    def __init__(self, dim, num_heads, num_kv_heads, ffn_dim):
        super().__init__()
        self.input_norm = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, num_kv_heads)
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.post_attn_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, ffn_dim)
        self.gate_mlp = nn.Parameter(torch.zeros(1))

    def forward(self, x, vision_states, cross_mask, text_row_mask):
        # cross_mask: [B, L, S]; text_row_mask: [B, L, 1]
        # 没有图像的行先放开 mask (防止 softmax 全 -inf 出 NaN), 再把输出置 0
        attn_out = self.cross_attn(self.input_norm(x), vision_states, cross_mask | ~text_row_mask)
        x = x + self.gate_attn.tanh() * attn_out * text_row_mask
        return x + self.gate_mlp.tanh() * self.mlp(self.post_attn_norm(x)) * text_row_mask


# ---------------------------------------------------------------------
# 完整的语言模型
# ---------------------------------------------------------------------
class MllamaTextModel(nn.Module):
    def __init__(self, vocab_size: int = 128256, dim: int = 4096, num_layers: int = 40, num_heads: int = 32,
                 num_kv_heads: int = 8, ffn_dim: int = 14336, rope_theta: float = 500000.0,
                 cross_attention_layers: List[int] = (3, 8, 13, 18, 23, 28, 33, 38)):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.rope_theta = rope_theta
        self.cross_attention_layers = set(cross_attention_layers)
        # 官方实现中词表额外多出 8 个位置, 用于 <|image|> 等新增的特殊 token
        self.embed_tokens = nn.Embedding(vocab_size + 8, dim)
        self.layers = nn.ModuleList([
            CrossAttentionDecoderLayer(dim, num_heads, num_kv_heads, ffn_dim) if i in self.cross_attention_layers
            else SelfAttentionDecoderLayer(dim, num_heads, num_kv_heads, ffn_dim)
            for i in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                vision_states: Optional[torch.Tensor] = None, cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        input_ids / attention_mask: [B, L]
        vision_states: [B, S, D] 投影后的视觉 token; 为 None 时是纯文本, 跳过所有 cross-attention 层
        cross_mask:    [B, L, S] 每个文本 token 能看到哪些视觉 token
        return: logits [B, L, vocab]
        """
        x = self.embed_tokens(input_ids)
        position_ids = (attention_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        cos, sin = rope_cos_sin(position_ids, self.head_dim, self.rope_theta)
        self_mask = causal_padding_mask(attention_mask)
        text_row_mask = cross_mask.any(dim=-1, keepdim=True) if cross_mask is not None else None

        for i, layer in enumerate(self.layers):
            if i in self.cross_attention_layers:
                if vision_states is not None:
                    x = layer(x, vision_states, cross_mask, text_row_mask)
            else:
                x = layer(x, cos, sin, self_mask)
        return self.lm_head(self.norm(x))
