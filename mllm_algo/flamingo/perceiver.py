"""Perceiver Resampler: 把数量不定的视觉特征 (多帧 × 多 patch) 压缩成固定数量 (默认 64) 的视觉 token。

一组可学习的 latent 作为 query, [视觉特征 ; latent] 拼接后作为 key/value 做 cross-attention。
和 BLIP-2 的 Q-Former 作用类似, 但只有视觉信息参与, 不和文本交互。
"""
import torch
import torch.nn as nn

from ..common import FeedForward, MultiHeadAttention


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)

    def forward(self, media: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        media = self.norm_media(media)
        latents = self.norm_latents(latents)
        # key/value 里也拼上 latent 自己, 让 latent 之间可以交换信息
        context = torch.cat([media, latents], dim=1)
        return self.attn(latents, context)


class PerceiverResampler(nn.Module):
    def __init__(self, dim: int, depth: int = 6, num_heads: int = 8, num_latents: int = 64, max_frames: int = 8):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.frame_embed = nn.Parameter(torch.randn(max_frames, 1, dim) * 0.02)  # 视频的时间位置编码
        self.layers = nn.ModuleList([
            nn.ModuleList([PerceiverAttention(dim, num_heads), nn.LayerNorm(dim), FeedForward(dim)])
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, media: torch.Tensor) -> torch.Tensor:
        """media: [B, T, N, D], T 为帧数 (单张图像 T = 1)。return: [B, num_latents, D]"""
        B, T, N, D = media.shape
        media = (media + self.frame_embed[:T]).reshape(B, T * N, D)  # 时间维和空间维展平

        latents = self.latents.expand(B, -1, -1)
        for attn, norm_ffn, ffn in self.layers:
            latents = latents + attn(media, latents)
            latents = latents + ffn(norm_ffn(latents))
        return self.norm(latents)
