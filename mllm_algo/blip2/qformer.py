"""Q-Former (Querying Transformer): BLIP-2 中连接冻结图像编码器和冻结 LLM 的桥梁。

输入序列 = [32 个可学习 query ; 文本 token], 每层结构:
    Self-Attn (query 和 text 共享, 通过 mask 控制谁能看到谁)
    Cross-Attn (只有 query 和图像特征交互, 每隔一层插一个)
    FFN (query 和 text 各一套)

query 是"信息瓶颈": 不管图像有多少 patch, 都只输出 32 个向量, 迫使它只保留和文本最相关的视觉信息。
结构上沿用 BERT 的 Post-Norm, 原版用 BERT-base 初始化。
"""
from typing import Optional

import torch
import torch.nn as nn

from ..common import FeedForward, MultiHeadAttention, causal_mask, padding_mask


class QFormerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, image_dim: int, has_cross_attn: bool):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads)
        self.norm_sa = nn.LayerNorm(dim)

        self.has_cross_attn = has_cross_attn
        if has_cross_attn:
            self.cross_attn = MultiHeadAttention(dim, num_heads, context_dim=image_dim)
            self.norm_ca = nn.LayerNorm(dim)

        self.ffn_query, self.norm_ffn_query = FeedForward(dim), nn.LayerNorm(dim)
        self.ffn_text, self.norm_ffn_text = FeedForward(dim), nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, num_query: int, mask: Optional[torch.Tensor],
                image_embeds: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.norm_sa(x + self.self_attn(x, mask=mask))

        query, text = x[:, :num_query], x[:, num_query:]
        if self.has_cross_attn and num_query > 0:
            # 只有 query 看图像; text 只能通过 self-attention 从 query 间接获取视觉信息
            query = self.norm_ca(query + self.cross_attn(query, image_embeds))

        query = self.norm_ffn_query(query + self.ffn_query(query))
        text = self.norm_ffn_text(text + self.ffn_text(text))
        return torch.cat([query, text], dim=1)


class QFormer(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 768, depth: int = 12, num_heads: int = 12,
                 image_dim: int = 1408, num_query: int = 32, max_len: int = 512, cross_attn_every: int = 2):
        super().__init__()
        self.dim = dim
        self.num_query = num_query
        self.query_tokens = nn.Parameter(torch.randn(1, num_query, dim) * 0.02)
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.embed_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)  # 只在 ITG 中使用
        self.layers = nn.ModuleList([
            QFormerLayer(dim, num_heads, image_dim, has_cross_attn=(i % cross_attn_every == 0))
            for i in range(depth)
        ])

    def forward(self, image_embeds: Optional[torch.Tensor] = None, input_ids: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None):
        """
        image_embeds: [B, N_patch, D_img], 为 None 时只编码文本
        input_ids:    [B, L], 为 None 时只有 query
        mask:         [B, N_q + L, N_q + L], 由 build_qformer_mask 构造
        return: (query 输出 [B, N_q, D], 文本输出 [B, L, D])
        """
        parts = []
        if image_embeds is not None:
            parts.append(self.query_tokens.expand(image_embeds.shape[0], -1, -1))
        if input_ids is not None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            parts.append(self.token_embed(input_ids) + self.pos_embed(positions))
        x = self.embed_norm(torch.cat(parts, dim=1))

        num_query = self.num_query if image_embeds is not None else 0
        for layer in self.layers:
            x = layer(x, num_query, mask, image_embeds)
        return x[:, :num_query], x[:, num_query:]


def build_qformer_mask(num_query: int, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """BLIP-2 三个预训练任务的区别就在 self-attention mask 上 (True = 可见)。

        mode="bidirectional"  (ITM)  query 和 text 全部互相可见
        mode="causal"         (ITG)  query 只看 query; text 看全部 query + 自己之前的 text
        ITC 是单模态 mask (query 和 text 互相看不到), 实现上等价于 query、text 各自单独前向一次

    attention_mask: [B, L] 文本 padding mask, return: [B, N_q + L, N_q + L]
    """
    B, L = attention_mask.shape
    N = num_query + L
    key_mask = torch.cat([attention_mask.new_ones(B, 1, num_query).bool(), padding_mask(attention_mask)], dim=2)
    mask = key_mask.expand(B, N, N).clone()

    if mode == "causal":
        mask[:, :num_query, num_query:] = False  # query 看不到 text, 否则生成时信息泄露
        mask[:, num_query:, num_query:] &= causal_mask(L, attention_mask.device)
    else:
        assert mode == "bidirectional", f"未知的 mode: {mode}"
    return mask
