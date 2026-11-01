"""Llama 3.2-Vision (11B / 90B), 对应 HuggingFace 的 MllamaForConditionalGeneration。

    图像 ──切 tile──> 视觉编码器 (ViT-H + 门控 global 层, 多层特征拼接 7680 维)
                         ──Linear──> 视觉 token (投影到文本维度 4096)
                                          │ 作为 cross-attention 的 K/V
    "<|image|> 问题" ──> Llama 3.1 [32 层 self-attn, 每 4 层后插 1 层门控 cross-attn] ──> 下一个 token

和其它算法的对比:
    - 像 Flamingo: 视觉信息通过门控 cross-attention 注入, 而不是像 LLaVA 那样拼进输入序列,
      所以高分辨率图像 (4 tile × 1601 token) 不会让文本序列变长
    - 不像 Flamingo: 没有 Perceiver Resampler, 直接用全部 patch 特征; 视觉编码器也参与训练
    - 训练时 Llama 3.1 完全冻结, 只训练视觉编码器、投影层和 cross-attention 层,
      纯文本输入时跳过 cross-attention 层, 和 Llama 3.1 输出完全一致
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..common import IGNORE_INDEX, ModelOutput, causal_lm_loss, freeze
from .text_model import MllamaTextModel
from .vision_encoder import MllamaVisionEncoder


def image_token_spans(input_ids: torch.Tensor, image_token_id: int) -> List[Tuple[int, int]]:
    """一条样本中每张图对应的文本区间 [start, end), 区间内的 token 可以看到这张图。

    规则 (和官方 processor 一致):
      - 每张图从自己的 <|image|> 开始, 到下一个 <|image|> 之前结束 (只看最近的图)
      - 连续出现的多个 <|image|> 视为一组, 组内每张图都对后续文本可见
        例: "<|image|><|image|> 比较这两张图" 中, 文本能同时看到两张图
    """
    positions = (input_ids == image_token_id).nonzero().flatten().tolist()
    if not positions:
        return []
    spans = [[start, end] for start, end in zip(positions, positions[1:] + [len(input_ids)])]

    last_end = spans[-1][1]
    for span in reversed(spans):
        if span[1] == span[0] + 1:  # 紧跟着下一个 <|image|>, 延长到这一组之后文本的结尾
            span[1] = last_end
        last_end = span[1]
    return [tuple(span) for span in spans]


def build_cross_attention_mask(input_ids: torch.Tensor, image_token_id: int, aspect_ratio_mask: torch.Tensor,
                               tokens_per_tile: int) -> torch.Tensor:
    """return: [B, L, N_img * T * tokens_per_tile], True = 该文本 token 可以看到该视觉 token。

    先算 "文本 token × 图像" 的可见性, 再乘上 "图像 × tile" 的有效性 (补齐的空 tile 不可见),
    最后展开到每个 tile 的所有 patch token。
    """
    B, L = input_ids.shape
    N, T = aspect_ratio_mask.shape[1:]
    text_to_image = torch.zeros(B, L, N, dtype=torch.bool, device=input_ids.device)
    for b in range(B):
        for image_index, (start, end) in enumerate(image_token_spans(input_ids[b], image_token_id)):
            text_to_image[b, start:end, image_index] = True

    mask = text_to_image[..., None] & aspect_ratio_mask.bool()[:, None]  # [B, L, N, T]
    return mask.repeat_interleave(tokens_per_tile, dim=-1).flatten(2)


class MllamaForConditionalGeneration(nn.Module):
    def __init__(self, vision_model: MllamaVisionEncoder, language_model: MllamaTextModel, image_token_id: int):
        super().__init__()
        self.vision_model = vision_model
        self.language_model = language_model
        self.image_token_id = image_token_id
        self.multi_modal_projector = nn.Linear(vision_model.output_dim, language_model.dim)  # 7680 -> 4096

        # 冻结 Llama 3.1, 只放开新插入的 cross-attention 层
        freeze(self.language_model).train()
        for i in self.language_model.cross_attention_layers:
            for p in self.language_model.layers[i].parameters():
                p.requires_grad = True

    def encode_images(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask) -> torch.Tensor:
        """return: [B, N_img * T * (P + 1), D_text] 的视觉 token"""
        vision_out = self.vision_model(pixel_values, aspect_ratio_ids, aspect_ratio_mask)  # [B, N, T, P+1, 7680]
        return self.multi_modal_projector(vision_out).flatten(1, 3)

    def _logits(self, input_ids, attention_mask, vision_states, aspect_ratio_mask):
        cross_mask = None
        if vision_states is not None:
            cross_mask = build_cross_attention_mask(input_ids, self.image_token_id, aspect_ratio_mask,
                                                    self.vision_model.num_patches + 1)
        return self.language_model(input_ids, attention_mask, vision_states, cross_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                pixel_values: Optional[torch.Tensor] = None, aspect_ratio_ids: Optional[torch.Tensor] = None,
                aspect_ratio_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> ModelOutput:
        """
        input_ids:  [B, L], 右 padding, 每张图在文本中有一个 <|image|> 占位 token (只占 1 个位置)
        pixel_values / aspect_ratio_ids / aspect_ratio_mask: 来自 image_processing.preprocess_images, 再加 batch 维
        labels:     不传时对除 padding 和 <|image|> 之外的所有 token 算 loss; SFT 时把 prompt 部分设为 -100
        """
        vision_states = None
        if pixel_values is not None:
            vision_states = self.encode_images(pixel_values, aspect_ratio_ids, aspect_ratio_mask)
        logits = self._logits(input_ids, attention_mask, vision_states, aspect_ratio_mask)

        if labels is None:
            labels = input_ids.masked_fill(attention_mask == 0, IGNORE_INDEX)
            labels = labels.masked_fill(input_ids == self.image_token_id, IGNORE_INDEX)
        return ModelOutput(loss=causal_lm_loss(logits, labels), logits=logits)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 pixel_values: Optional[torch.Tensor] = None, aspect_ratio_ids: Optional[torch.Tensor] = None,
                 aspect_ratio_mask: Optional[torch.Tensor] = None, max_new_tokens: int = 20,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """贪心解码, 只返回新生成的 token。图像只编码一次 (真实实现中 cross-attn 的 K/V 也只算一次并缓存)。"""
        vision_states = None
        if pixel_values is not None:
            vision_states = self.encode_images(pixel_values, aspect_ratio_ids, aspect_ratio_mask)

        generated = []
        for _ in range(max_new_tokens):
            logits = self._logits(input_ids, attention_mask, vision_states, aspect_ratio_mask)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            if eos_token_id is not None and (torch.cat(generated, dim=1) == eos_token_id).any(dim=1).all():
                break
        return torch.cat(generated, dim=1)
