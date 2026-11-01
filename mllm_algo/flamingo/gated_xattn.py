"""GATED XATTN-DENSE: 插在冻结 LM 层之间的门控交叉注意力层。

    y = x + tanh(α_attn) · CrossAttn(x, 视觉 token)
    y = y + tanh(α_ffn)  · FFN(y)

α 初始化为 0, 训练开始时 tanh(α) = 0, 整个模型等价于原始 LM, 保证训练稳定、不破坏语言能力。
"""
import torch
import torch.nn as nn

from ..common import FeedForward, MultiHeadAttention


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, visual_dim: int, num_heads: int):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, context_dim=visual_dim)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)
        self.ffn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, visual_tokens: torch.Tensor, media_mask: torch.Tensor) -> torch.Tensor:
        """
        x:             [B, L, D] 文本隐状态
        visual_tokens: [B, N_img * num_latents, D_v] 所有图像的视觉 token
        media_mask:    [B, L, N_img * num_latents] 每个文本 token 能看到哪些视觉 token
        """
        # 前面还没出现图像的 token 不接收视觉信息。先放开这些行的 mask 防止 softmax 出 NaN, 再把输出置 0
        has_media = media_mask.any(dim=-1, keepdim=True)  # [B, L, 1]
        attn_out = self.attn(self.norm_attn(x), visual_tokens, mask=media_mask | ~has_media)
        attn_out = attn_out.masked_fill(~has_media, 0.0)

        x = x + self.attn_gate.tanh() * attn_out
        x = x + self.ffn_gate.tanh() * self.ffn(self.norm_ffn(x))
        return x


def build_media_mask(input_ids: torch.Tensor, media_token_id: int, num_images: int, num_latents: int) -> torch.Tensor:
    """每个文本 token 只关注它前面最近的那一张图像 (而不是之前所有图像)。

    例: "<image> 一只猫 <image> 一只狗" 中, "一只狗" 只能看到第 2 张图。
    return: [B, L, num_images * num_latents]
    """
    # text_time[b, i] = 第 i 个 token 之前 (含自己) 出现了几张图, 即它对应第几张图 (0 表示前面没有图)
    text_time = (input_ids == media_token_id).long().cumsum(dim=1)  # [B, L]
    # media_time[j] = 第 j 个视觉 token 属于第几张图 (从 1 开始)
    media_time = torch.arange(1, num_images + 1, device=input_ids.device).repeat_interleave(num_latents)
    return text_time[:, :, None] == media_time[None, None, :]
