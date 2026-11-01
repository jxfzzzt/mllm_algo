"""Flamingo: a Visual Language Model for Few-Shot Learning (Alayrac et al., 2022)

    图像/视频 ──冻结的视觉编码器──> Perceiver Resampler ──> 64 个视觉 token / 每张图
                                                                │ (作为 cross-attention 的 K/V)
    交错的图文序列 "<image> 描述1 <EOC> <image> 描述2 ..." ──> [GATED XATTN ─> 冻结的 LM 层] × N ──> 下一个 token

核心设计:
    1. 视觉编码器和 LM 都冻结, 只训练 Perceiver Resampler 和新插入的 GATED XATTN-DENSE 层
    2. 视觉信息通过 cross-attention 注入 (不像 BLIP-2 / LLaVA 那样拼到输入序列里), 序列长度不随图像数增加
    3. 每个文本 token 只看它前面最近的一张图, 支持任意交错的多图输入, 因此天然支持 few-shot in-context learning

论文中 <image> 和 <|endofchunk|> 两个新 token 的词向量是可训练的, 这里为了简单直接复用冻结的词表。
"""
from typing import Optional

import torch
import torch.nn as nn

from ..common import (IGNORE_INDEX, CausalLM, ModelOutput, VisionTransformer, causal_lm_loss,
                      causal_padding_mask, freeze)
from .gated_xattn import GatedCrossAttentionBlock, build_media_mask
from .perceiver import PerceiverResampler


class Flamingo(nn.Module):
    def __init__(self, vision_encoder: VisionTransformer, lang_model: CausalLM, media_token_id: int,
                 cross_attn_every: int = 1, perceiver_depth: int = 6, num_latents: int = 64, num_heads: int = 8):
        super().__init__()
        self.visual = freeze(vision_encoder)
        self.lang_model = freeze(lang_model)
        self.media_token_id = media_token_id
        self.perceiver = PerceiverResampler(vision_encoder.dim, perceiver_depth, num_heads, num_latents)

        # 每隔 cross_attn_every 层 LM, 在它前面插一个门控交叉注意力层 (key 为 LM 层的下标)
        lm_dim = lang_model.config.hidden_size
        self.xattn_layers = nn.ModuleDict({
            str(i): GatedCrossAttentionBlock(lm_dim, vision_encoder.dim, num_heads)
            for i in range(len(lang_model.layers)) if i % cross_attn_every == 0
        })

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, N_img, 3, H, W] (每条样本可以有多张图)。return: [B, N_img * num_latents, D_v]"""
        B, num_images = images.shape[:2]
        with torch.no_grad():
            patch_embeds = self.visual(images.flatten(0, 1))[:, 1:]  # 去掉 [CLS], [B * N_img, N_patch, D_v]
        latents = self.perceiver(patch_embeds.unsqueeze(1))  # 每张图独立重采样, 帧数 T = 1
        return latents.reshape(B, num_images * self.perceiver.num_latents, -1)

    def _lm_forward(self, visual_tokens: torch.Tensor, num_images: int,
                    input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """在冻结 LM 的每一层之前插入 GATED XATTN, 返回 logits。"""
        lm = self.lang_model
        media_mask = build_media_mask(input_ids, self.media_token_id, num_images, self.perceiver.num_latents)
        x = lm.embed(lm.embed_tokens(input_ids), attention_mask)
        self_attn_mask = causal_padding_mask(attention_mask)

        for i, layer in enumerate(lm.layers):
            if str(i) in self.xattn_layers:
                x = self.xattn_layers[str(i)](x, visual_tokens, media_mask)
            x = layer(x, mask=self_attn_mask)
        return lm.lm_head(lm.norm(x))

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutput:
        """input_ids 中每个 <image> token 对应 images 里按顺序排列的一张图, 右 padding。"""
        visual_tokens = self.encode_images(images)
        logits = self._lm_forward(visual_tokens, images.shape[1], input_ids, attention_mask)

        # padding 和 <image> 占位符不算 loss
        labels = input_ids.masked_fill(attention_mask == 0, IGNORE_INDEX)
        labels = labels.masked_fill(input_ids == self.media_token_id, IGNORE_INDEX)
        return ModelOutput(loss=causal_lm_loss(logits, labels), logits=logits)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 max_new_tokens: int = 20, eos_token_id: Optional[int] = None) -> torch.Tensor:
        """贪心解码, 只返回新生成的 token。few-shot 时把示例图文对按顺序拼在 query 前面即可。"""
        visual_tokens = self.encode_images(images)  # 图像只编码一次
        generated = []
        for _ in range(max_new_tokens):
            logits = self._lm_forward(visual_tokens, images.shape[1], input_ids, attention_mask)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            if eos_token_id is not None and (torch.cat(generated, dim=1) == eos_token_id).any(dim=1).all():
                break
        return torch.cat(generated, dim=1)
