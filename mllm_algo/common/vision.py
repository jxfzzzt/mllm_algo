"""Vision Transformer (ViT), 所有算法共用的图像编码器。

实际项目中会换成预训练权重: CLIP ViT-L/14 (CLIP / LLaVA / Flamingo 用 NFNet), EVA ViT-g (BLIP-2) 等。
"""
import torch
import torch.nn as nn

from .attention import TransformerBlock


class VisionTransformer(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, dim: int = 768,
                 depth: int = 12, num_heads: int = 12):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, images: torch.Tensor, layer: int = -1) -> torch.Tensor:
        """
        images: [B, 3, H, W]
        layer:  取第几层的输出, -1 是最后一层 (会过最终 LayerNorm), -2 是倒数第二层 (LLaVA / BLIP-2 的做法)
        return: [B, 1 + N_patch, D], 第 0 个位置是 [CLS]
        """
        x = self.patch_embed(images).flatten(2).transpose(1, 2)  # [B, N, D]
        cls = self.cls_token.expand(x.shape[0], -1, -1) # expand 在进行复制的时候是共享内存的，节省显存
        x = torch.cat([cls, x], dim=1) + self.pos_embed

        num_blocks = len(self.blocks) + layer + 1 if layer < 0 else layer
        for block in self.blocks[:num_blocks]:
            x = block(x)
        return self.norm(x) if num_blocks == len(self.blocks) else x
