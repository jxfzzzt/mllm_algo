"""CLIP: Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)

核心思想: 图像编码器和文本编码器各自把输入映射到同一个向量空间, 在一个 batch 内做对比学习,
         让匹配的图文对相似度高, 不匹配的低。训练完可以直接用文本 prompt 做零样本分类。

    image ──ViT──> [CLS] ──Linear──> L2 norm ─┐
                                              ├── 余弦相似度 × logit_scale ──> 对称交叉熵
    text  ──Transformer──> [EOS] ──Linear──> L2 norm ─┘
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import ModelOutput, VisionTransformer, contrastive_loss
from .text_encoder import CLIPTextEncoder


class CLIP(nn.Module):
    def __init__(self, vision_encoder: VisionTransformer, text_encoder: CLIPTextEncoder, embed_dim: int = 512):
        super().__init__()
        self.visual = vision_encoder
        self.text = text_encoder
        self.image_proj = nn.Linear(vision_encoder.dim, embed_dim, bias=False)
        self.text_proj = nn.Linear(text_encoder.dim, embed_dim, bias=False)
        # 可学习的温度系数, 存的是 log(1 / τ), 初始 τ = 0.07
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        cls = self.visual(images)[:, 0]
        return F.normalize(self.image_proj(cls), dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        eos = self.text(input_ids, attention_mask)
        return F.normalize(self.text_proj(eos), dim=-1)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutput:
        image_feat = self.encode_image(images)  # [B, E]
        text_feat = self.encode_text(input_ids, attention_mask)  # [B, E]

        # 温度上限 100, 防止训练不稳定 (多卡训练时这里会先 all_gather 所有卡的特征, 扩大负样本数)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_feat @ text_feat.t()  # [B, B]
        return ModelOutput(loss=contrastive_loss(logits_per_image), logits=logits_per_image)

    @torch.no_grad()
    def zero_shot_classify(self, images: torch.Tensor, class_input_ids: torch.Tensor,
                           class_attention_mask: torch.Tensor) -> torch.Tensor:
        """零样本分类: 每个类别构造一个 prompt (如 "a photo of a {class}"), 选和图像最相似的那个。

        class_input_ids: [num_classes, L], return: [B, num_classes] 的概率
        """
        image_feat = self.encode_image(images)
        class_feat = self.encode_text(class_input_ids, class_attention_mask)
        return (self.logit_scale.exp() * image_feat @ class_feat.t()).softmax(dim=-1)
