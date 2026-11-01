"""Llama 3.2-Vision 的图像编码器: ViT-H/14 + 8 层门控 global 层 + 多层特征拼接。

流程 (以 11B 为例, D = 1280, 每个 tile 40×40 = 1600 个 patch):
    tiles ──patch embed──> + pre-tile 位置编码 ──> + [CLS] ──> + patch 位置编码 & tile 位置编码
          ──> 32 层 ViT (同一张图的所有 tile 拼成一个序列做 attention, 并记录中间层输出)
          ──> + post-tile 位置编码 ──> 8 层门控 global Transformer
          ──> 拼接 [最终输出, 第 3/7/15/23/30 层输出] ──> 每个 token 维度 1280 × 6 = 7680

关键设计:
    - tile 位置编码按 aspect_ratio_id 查表, 让模型知道每个 tile 在原图中的位置
    - 中间层特征保留了更多细粒度信息 (浅层偏纹理细节, 深层偏语义), 拼接后一起送给语言模型
    - global 层是在预训练 ViT 之上新加的, 用门控逐步引入, 不破坏预训练特征
"""
from typing import Sequence

import torch
import torch.nn as nn

from ..common import TransformerBlock


class GatedTransformerBlock(TransformerBlock):
    """attention 和 FFN 的输出都乘上 tanh(gate)。gate 初始化为 0, 训练开始时是恒等映射。"""

    def __init__(self, dim: int, num_heads: int):
        super().__init__(dim, num_heads)
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        x = x + self.gate_attn.tanh() * self.self_attn(self.norm_sa(x), mask=mask)
        return x + self.gate_ffn.tanh() * self.ffn(self.norm_ffn(x))


class TilePositionEmbedding(nn.Module):
    """按宽高比查表得到每个 tile 的位置编码, 同一个 tile 内所有 patch 共享。门控加到输入上。"""

    def __init__(self, num_aspect_ratios: int, max_num_tiles: int, dim: int):
        super().__init__()
        self.max_num_tiles, self.dim = max_num_tiles, dim
        self.embedding = nn.Embedding(num_aspect_ratios + 1, max_num_tiles * dim)  # id 0 是 padding
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, T, P, D], aspect_ratio_ids: [B]
        emb = self.embedding(aspect_ratio_ids).view(-1, self.max_num_tiles, 1, self.dim)
        return x + self.gate.tanh() * emb[:, :x.shape[1]]


class GatedPositionalEmbedding(nn.Module):
    """patch 位置编码 (所有 tile 共享) 和 "tile × patch" 位置编码 (按宽高比查表) 的门控插值:

        x + (1 - tanh(gate)) · patch_pos + tanh(gate) · tile_patch_pos
    """

    def __init__(self, num_aspect_ratios: int, max_num_tiles: int, num_tokens: int, dim: int):
        super().__init__()
        self.max_num_tiles, self.num_tokens, self.dim = max_num_tiles, num_tokens, dim
        self.patch_pos = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.tile_patch_pos = nn.Embedding(num_aspect_ratios + 1, max_num_tiles * num_tokens * dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        gate = self.gate.tanh()
        tile_patch_pos = self.tile_patch_pos(aspect_ratio_ids).view(-1, self.max_num_tiles, self.num_tokens, self.dim)
        return x + (1 - gate) * self.patch_pos + gate * tile_patch_pos[:, :x.shape[1]]


class MllamaVisionEncoder(nn.Module):
    def __init__(self, image_size: int = 560, patch_size: int = 14, dim: int = 1280, num_heads: int = 16,
                 num_layers: int = 32, num_global_layers: int = 8, max_num_tiles: int = 4,
                 num_aspect_ratios: int = 8, intermediate_layers: Sequence[int] = (3, 7, 15, 23, 30)):
        super().__init__()
        self.dim = dim
        self.num_patches = (image_size // patch_size) ** 2
        self.intermediate_layers = set(intermediate_layers)
        self.output_dim = dim * (1 + len(intermediate_layers))  # 11B: 1280 × 6 = 7680
        num_tokens = self.num_patches + 1  # 每个 tile 有自己的 [CLS]

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.pre_tile_pos_embed = TilePositionEmbedding(num_aspect_ratios, max_num_tiles, dim)
        self.pos_embed = GatedPositionalEmbedding(num_aspect_ratios, max_num_tiles, num_tokens, dim)
        self.post_tile_pos_embed = TilePositionEmbedding(num_aspect_ratios, max_num_tiles, dim)
        self.ln_pre = nn.LayerNorm(dim)
        self.ln_post = nn.LayerNorm(dim)

        self.transformer = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(num_layers)])
        self.global_transformer = nn.ModuleList([GatedTransformerBlock(dim, num_heads) for _ in range(num_global_layers)])

    def forward(self, pixel_values: torch.Tensor, aspect_ratio_ids: torch.Tensor,
                aspect_ratio_mask: torch.Tensor) -> torch.Tensor:
        """
        pixel_values:      [B, N_img, T, 3, H, W]   T = max_num_tiles
        aspect_ratio_ids:  [B, N_img]
        aspect_ratio_mask: [B, N_img, T]           1 = 真实 tile
        return:            [B, N_img, T, P + 1, output_dim]
        """
        B, N, T = pixel_values.shape[:3]
        P, D = self.num_patches, self.dim
        ratio_ids = aspect_ratio_ids.flatten()  # [B*N]

        # 1. patch embedding + 位置编码, 每张图处理成 [T, P+1, D]
        x = self.patch_embed(pixel_values.flatten(0, 2)).flatten(2).transpose(1, 2)  # [B*N*T, P, D]
        x = self.pre_tile_pos_embed(x.view(B * N, T, P, D), ratio_ids)
        x = torch.cat([self.cls_token.expand(B * N, T, 1, D), x], dim=2)  # [B*N, T, P+1, D]
        x = self.ln_pre(self.pos_embed(x, ratio_ids))

        # 2. 同一张图的所有 tile 拼成一个长序列, 补齐的空 tile 不可见
        x = x.view(B * N, T * (P + 1), D)
        key_mask = aspect_ratio_mask.flatten(0, 1).bool().repeat_interleave(P + 1, dim=1)  # [B*N, T*(P+1)]
        key_mask[:, :P + 1] = True  # 第一个 tile 始终可见, 防止 batch 里补齐的空图整行被 mask 出 NaN
        mask = key_mask[:, None, :]

        # 3. 32 层 ViT, 记录中间层输出
        intermediates = []
        for i, block in enumerate(self.transformer):
            x = block(x, mask=mask)
            if i in self.intermediate_layers:
                intermediates.append(x)
        x = self.ln_post(x)

        # 4. post-tile 位置编码 + 8 层门控 global Transformer
        x = self.post_tile_pos_embed(x.view(B * N, T, P + 1, D), ratio_ids).view(B * N, T * (P + 1), D)
        for block in self.global_transformer:
            x = block(x, mask=mask)

        # 5. 最终输出和中间层特征在通道维拼接
        x = torch.cat([x] + intermediates, dim=-1)
        return x.view(B, N, T, P + 1, self.output_dim)
