"""ALBEF: Align before Fuse: Vision and Language Representation Learning with Momentum Distillation (Li et al., 2021)

    image ──ViT──> image_embeds ──[CLS]──> vision_proj ─┐
                       │                                ├── ITC (对齐)
    text ──BERT 前 6 层──> text_embeds ──[CLS]──> text_proj ─┘
                       │
                       └──> BERT 后 6 层 (cross-attn 看 image_embeds) ──> ITM / MLM (融合)

预训练目标:
    ITC  图文对比学习, 在融合之前先对齐单模态特征; 用动量编码器 + 特征队列提供大量负样本
    ITM  图文匹配二分类, 基于 ITC 相似度采样难负样本
    MLM  借助图像预测被 mask 的词

动量蒸馏 (Momentum Distillation): 网络图文对噪声很大, 有些"负样本"其实也匹配, 被 mask 的词也可能有多个合理答案。
    所以用动量模型 (主模型参数的 EMA) 的预测作为软标签, 和 one-hot 硬标签按 alpha 混合, ITC 和 MLM 都用。

BLIP 沿用了 ALBEF 的 ITC (动量蒸馏 + 队列) 和 ITM (难负样本), 把 MLM 换成了 LM, 把拆分的 BERT 换成了 MED。
"""
import copy
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import IGNORE_INDEX, VisionTransformer, sample_hard_negatives
from .masking import mask_tokens
from .text_encoder import ALBEFTextEncoder


@dataclass
class ALBEFOutput:
    loss: torch.Tensor
    loss_itc: torch.Tensor
    loss_itm: torch.Tensor
    loss_mlm: torch.Tensor


class ALBEFPretrain(nn.Module):
    def __init__(self, vision_encoder: VisionTransformer, text_encoder: ALBEFTextEncoder, mask_token_id: int,
                 special_token_ids: Sequence[int] = (), embed_dim: int = 256, queue_size: int = 65536,
                 momentum: float = 0.995, alpha: float = 0.4, mlm_prob: float = 0.15):
        super().__init__()
        self.visual = vision_encoder
        self.text = text_encoder
        self.vision_proj = nn.Linear(vision_encoder.dim, embed_dim)
        self.text_proj = nn.Linear(text_encoder.dim, embed_dim)
        self.itm_head = nn.Linear(text_encoder.dim, 2)
        self.temp = nn.Parameter(torch.tensor(0.07))
        self.mask_token_id = mask_token_id
        self.special_token_ids = tuple(special_token_ids)
        self.mlm_prob = mlm_prob

        # ---- 动量模型: 参数是主模型的指数滑动平均, 不参与反向传播 ----
        self.momentum = momentum
        self.alpha = alpha  # 软标签所占的权重
        self.visual_m = copy.deepcopy(self.visual)
        self.vision_proj_m = copy.deepcopy(self.vision_proj)
        self.text_m = copy.deepcopy(self.text)
        self.text_proj_m = copy.deepcopy(self.text_proj)
        self.model_pairs = [
            (self.visual, self.visual_m),
            (self.vision_proj, self.vision_proj_m),
            (self.text, self.text_m),
            (self.text_proj, self.text_proj_m),
        ]
        for _, model_m in self.model_pairs:
            for p in model_m.parameters():
                p.requires_grad = False

        # ---- 特征队列: 存放历史 batch 的动量特征, 作为 ITC 的额外负样本 ----
        self.queue_size = queue_size
        self.register_buffer("image_queue", F.normalize(torch.randn(embed_dim, queue_size), dim=0))
        self.register_buffer("text_queue", F.normalize(torch.randn(embed_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                alpha: Optional[float] = None) -> ALBEFOutput:
        """
        input_ids 第一个 token 是 [CLS], 右 padding。
        alpha: 动量蒸馏权重, 默认用 self.alpha。论文在第一个 epoch 内把它从 0 线性升到 0.4,
               因为训练初期动量模型还很差, 它给出的软标签不可信。
        """
        alpha = self.alpha if alpha is None else alpha
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
            self._momentum_update()
            image_embeds_m = self.visual_m(images)

        image_embeds = self.visual(images)  # [B, 1+N, D_v]
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0]), dim=-1)
        text_embeds = self.text.encode_text(input_ids, attention_mask)  # [B, L, D], 只经过前 6 层
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0]), dim=-1)

        loss_itc, sim_i2t = self._itc_loss(image_feat, text_feat, image_embeds_m, input_ids, attention_mask, alpha)
        loss_itm = self._itm_loss(image_embeds, text_embeds, attention_mask, sim_i2t)
        loss_mlm = self._mlm_loss(image_embeds, image_embeds_m, input_ids, attention_mask, alpha)
        return ALBEFOutput(loss=loss_itc + loss_itm + loss_mlm, loss_itc=loss_itc, loss_itm=loss_itm, loss_mlm=loss_mlm)

    # ------------------------------------------------------------------
    # ITC: 图文对比学习 + 动量蒸馏
    # ------------------------------------------------------------------
    def _itc_loss(self, image_feat, text_feat, image_embeds_m, input_ids, attention_mask, alpha):
        with torch.no_grad():
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0]), dim=-1)
            text_feat_m = F.normalize(self.text_proj_m(self.text_m.encode_text(input_ids, attention_mask)[:, 0]), dim=-1)
            # 当前 batch 的动量特征 + 队列一起作为候选: [E, B + Q]
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone()], dim=1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone()], dim=1)

            # 软标签 = alpha · 动量模型的相似度分布 + (1 - alpha) · one-hot
            hard_targets = torch.zeros(image_feat.shape[0], image_feat_all.shape[1], device=image_feat.device)
            hard_targets.fill_diagonal_(1)
            i2t_targets = alpha * F.softmax(image_feat_m @ text_feat_all / self.temp, dim=1) + (1 - alpha) * hard_targets
            t2i_targets = alpha * F.softmax(text_feat_m @ image_feat_all / self.temp, dim=1) + (1 - alpha) * hard_targets

        sim_i2t = image_feat @ text_feat_all / self.temp  # [B, B + Q]
        sim_t2i = text_feat @ image_feat_all / self.temp
        loss_i2t = -(F.log_softmax(sim_i2t, dim=1) * i2t_targets).sum(dim=1).mean()
        loss_t2i = -(F.log_softmax(sim_t2i, dim=1) * t2i_targets).sum(dim=1).mean()

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        B = image_feat.shape[0]
        return (loss_i2t + loss_t2i) / 2, sim_i2t[:, :B]

    # ------------------------------------------------------------------
    # ITM: 融合编码器 + 难负样本二分类
    # ------------------------------------------------------------------
    def _itm_loss(self, image_embeds, text_embeds, attention_mask, sim_i2t):
        B = text_embeds.shape[0]
        neg_image_index, neg_text_index = sample_hard_negatives(sim_i2t)

        # 正样本 B 对 + (难负图, 文) B 对 + (图, 难负文) B 对。
        # 融合编码器的输入直接复用前 6 层的 text_embeds, 不用重新编码文本
        images_all = torch.cat([image_embeds, image_embeds[neg_image_index], image_embeds], dim=0)
        texts_all = torch.cat([text_embeds, text_embeds, text_embeds[neg_text_index]], dim=0)
        mask_all = torch.cat([attention_mask, attention_mask, attention_mask[neg_text_index]], dim=0)

        fused = self.text.fuse(texts_all, mask_all, images_all)
        logits = self.itm_head(fused[:, 0])  # 用融合后 [CLS] 的输出做二分类
        labels = torch.cat([torch.ones(B), torch.zeros(2 * B)]).long().to(logits.device)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # MLM: 借助图像的掩码语言模型 + 动量蒸馏
    # ------------------------------------------------------------------
    def _mlm_loss(self, image_embeds, image_embeds_m, input_ids, attention_mask, alpha):
        masked_ids, labels = mask_tokens(input_ids, attention_mask, self.text.vocab_size, self.mask_token_id,
                                         self.special_token_ids, self.mlm_prob)
        logits = self.text.mlm_logits(masked_ids, attention_mask, image_embeds)  # [B, L, V]

        selected = labels != IGNORE_INDEX
        if not selected.any():  # 极小概率一个词都没被 mask
            return logits.sum() * 0.0
        logits, labels = logits[selected], labels[selected]  # 只在被 mask 的位置算 loss

        with torch.no_grad():
            soft_labels = F.softmax(self.text_m.mlm_logits(masked_ids, attention_mask, image_embeds_m)[selected], dim=-1)

        loss_hard = F.cross_entropy(logits, labels)
        loss_soft = -(F.log_softmax(logits, dim=-1) * soft_labels).sum(dim=-1).mean()
        return (1 - alpha) * loss_hard + alpha * loss_soft

    # ------------------------------------------------------------------
    # 动量模型 & 队列
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _momentum_update(self):
        for model, model_m in self.model_pairs:
            for p, p_m in zip(model.parameters(), model_m.parameters()):
                p_m.data.mul_(self.momentum).add_(p.data, alpha=1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat: torch.Tensor, text_feat: torch.Tensor):
        B = image_feat.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % B == 0, "queue_size 需要能被 batch size 整除"
        self.image_queue[:, ptr:ptr + B] = image_feat.t()
        self.text_queue[:, ptr:ptr + B] = text_feat.t()
        self.queue_ptr[0] = (ptr + B) % self.queue_size
