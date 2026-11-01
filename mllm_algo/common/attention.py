"""通用的 Transformer 组件: 多头注意力、前馈网络、Transformer Block 和 mask 工具函数。

约定: 所有 mask 都是 bool 张量, True 表示"可以看到", 形状可广播到 [B, L_q, L_k]。
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头注意力。context 为 None 时是 self-attention, 否则是 cross-attention。"""

    def __init__(self, dim: int, num_heads: int, context_dim: Optional[int] = None):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        context_dim = context_dim or dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim)
        self.v_proj = nn.Linear(context_dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L_q, D], context: [B, L_k, D_ctx], mask: [B, L_q, L_k]
        context = x if context is None else context
        B, L_q, _ = x.shape
        L_k = context.shape[1]

        q = self.q_proj(x).view(B, L_q, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(context).view(B, L_k, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(context).view(B, L_k, self.num_heads, -1).transpose(1, 2)

        attn_mask = None if mask is None else mask.unsqueeze(1)  # 广播到所有 head
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, L_q, -1))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block: Self-Attn -> (可选) Cross-Attn -> FFN。"""

    def __init__(self, dim: int, num_heads: int, context_dim: Optional[int] = None):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads)

        self.has_cross_attn = context_dim is not None
        if self.has_cross_attn:
            self.norm_ca = nn.LayerNorm(dim)
            self.cross_attn = MultiHeadAttention(dim, num_heads, context_dim)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm_sa(x), mask=mask)
        if self.has_cross_attn and context is not None:
            x = x + self.cross_attn(self.norm_ca(x), context, mask=context_mask)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------
# mask 工具函数
# ---------------------------------------------------------------------
def padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """[B, L] 的 0/1 padding mask -> [B, 1, L] 的 key mask (padding 位置不可见)。"""
    return attention_mask.bool().unsqueeze(1)


def causal_mask(length: int, device=None) -> torch.Tensor:
    """[L, L] 下三角 mask, 每个位置只能看到自己和之前的位置。"""
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))


def causal_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """因果 mask 和 padding mask 的组合, 返回 [B, L, L]。

    额外保证每个位置都能看到自己, 避免左 padding 时整行全被 mask 导致 softmax 出 NaN。
    """
    L = attention_mask.shape[1]
    eye = torch.eye(L, dtype=torch.bool, device=attention_mask.device)
    return (causal_mask(L, attention_mask.device) & padding_mask(attention_mask)) | eye
