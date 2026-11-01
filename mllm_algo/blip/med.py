"""BLIP 的 MED (Multimodal mixture of Encoder-Decoder)。

同一个文本 Transformer 通过 mode 切换成三种功能, 除了 self-attention 之外的参数全部共享:

    mode="text"        双向 SA                    + FFN    单模态文本编码器         -> ITC
    mode="multimodal"  双向 SA       + CA(图像)   + FFN    图像引导的文本编码器     -> ITM
    mode="decoder"     因果 SA(独立) + CA(图像)   + FFN    图像引导的文本解码器     -> LM

编码和解码的差异主要体现在 SA 上 (双向 vs 因果), 所以 decoder 单独一套 SA, 其余共享以提高训练效率。
"""
from typing import Optional

import torch
import torch.nn as nn

from ..common import FeedForward, MultiHeadAttention, causal_padding_mask, padding_mask

MODES = ("text", "multimodal", "decoder")


class MEDLayer(nn.Module):
    """各模块被哪些任务使用:

        模块                          ITC (text)   ITM (multimodal)   LM (decoder)
        norm_sa + self_attn  (双向)       ✓              ✓
        norm_sa_dec + self_attn_dec (因果)                                 ✓
        norm_ca + cross_attn                            ✓                  ✓
        norm_ffn + ffn                    ✓              ✓                  ✓
    """

    def __init__(self, dim: int, num_heads: int, image_dim: int):
        super().__init__()
        # ITC + ITM: 编码器的双向 SA, 每个 token 都能看到整句话
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads)
        # LM: 解码器的因果 SA, 只能看左边; 编码和解码的差异主要在 SA, 所以单独一套不共享
        self.norm_sa_dec = nn.LayerNorm(dim)
        self.self_attn_dec = MultiHeadAttention(dim, num_heads)
        # ITM + LM: 文本 query 去看图像 patch; ITC 要单模态编码文本, 不用这一层
        self.norm_ca = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads, context_dim=image_dim)
        # ITC + ITM + LM: 三种模式全部共享
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, mode: str,
                image_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mode == "decoder":  # LM
            x = x + self.self_attn_dec(self.norm_sa_dec(x), mask=mask)
        else:  # ITC / ITM
            x = x + self.self_attn(self.norm_sa(x), mask=mask)

        if mode != "text":  # ITM / LM 才注入图像信息
            x = x + self.cross_attn(self.norm_ca(x), image_embeds)

        return x + self.ffn(self.norm_ffn(x))  # 三种模式都走


class MED(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 768, depth: int = 12, num_heads: int = 12,
                 image_dim: int = 768, max_len: int = 512):
        super().__init__()
        self.dim = dim
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList([MEDLayer(dim, num_heads, image_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)  # 只有 decoder 模式会用到

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, mode: str = "text",
                image_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """return: 每个 token 的隐状态 [B, L, D]"""
        assert mode in MODES, f"mode 必须是 {MODES} 之一"
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        mask = causal_padding_mask(attention_mask) if mode == "decoder" else padding_mask(attention_mask)
        for layer in self.layers:
            x = layer(x, mask, mode, image_embeds)
        return self.norm(x)
