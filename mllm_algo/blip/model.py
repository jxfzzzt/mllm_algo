"""BLIP: Bootstrapping Language-Image Pre-training (Li et al., 2022)

模型结构: ViT 图像编码器 + MED 文本模型 (见 med.py), 全部参数一起训练。
预训练目标 (每个 batch 同时计算三个 loss):
    ITC  图文对比学习, 沿用 ALBEF 的动量编码器 + 特征队列 + 动量蒸馏 (软标签)
    ITM  图文匹配二分类, 基于 ITC 相似度采样难负样本
    LM   以图像为条件的自回归文本生成, 是 BLIP 相对 ALBEF 新增的, 让模型具备生成能力

数据层面的 CapFilt (用 Captioner 生成合成描述 + 用 Filter 过滤噪声图文对) 是用训练好的
LM 头和 ITM 头对网络数据做清洗, 不涉及新的模型结构, 这里不实现。
"""
import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import IGNORE_INDEX, VisionTransformer, causal_lm_loss, sample_hard_negatives
from .med import MED


@dataclass
class BLIPOutput:
    loss: torch.Tensor
    loss_itc: torch.Tensor
    loss_itm: torch.Tensor
    loss_lm: torch.Tensor


class BLIPPretrain(nn.Module):
    def __init__(self, vision_encoder: VisionTransformer, text_model: MED, enc_token_id: int, dec_token_id: int,
                 embed_dim: int = 256, queue_size: int = 57600, momentum: float = 0.995, alpha: float = 0.4):
        super().__init__()
        self.visual = vision_encoder
        self.text = text_model
        self.vision_proj = nn.Linear(vision_encoder.dim, embed_dim)
        self.text_proj = nn.Linear(text_model.dim, embed_dim)
        self.itm_head = nn.Linear(text_model.dim, 2)
        self.temp = nn.Parameter(torch.tensor(0.07))
        self.enc_token_id = enc_token_id  # ITM 时把 [CLS] 换成 [ENC]
        self.dec_token_id = dec_token_id  # LM  时把 [CLS] 换成 [DEC]

        # ---- 动量编码器: 参数是主模型的指数滑动平均, 不参与反向传播 ----
        self.momentum = momentum
        self.alpha = alpha  # 软标签 (动量模型预测) 所占的权重
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

        # ---- 特征队列: 存放历史 batch 的动量特征, 大幅增加 ITC 的负样本数 ----
        self.queue_size = queue_size
        self.register_buffer("image_queue", F.normalize(torch.randn(embed_dim, queue_size), dim=0))
        self.register_buffer("text_queue", F.normalize(torch.randn(embed_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BLIPOutput:
        """input_ids 第一个 token 是 [CLS], 右 padding。"""
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual(images)  # [B, 1+N, D_v]
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0]), dim=-1)
        text_embeds = self.text(input_ids, attention_mask, mode="text")
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0]), dim=-1)

        loss_itc, sim_i2t = self._itc_loss(images, input_ids, attention_mask, image_feat, text_feat)
        loss_itm = self._itm_loss(image_embeds, input_ids, attention_mask, sim_i2t)
        loss_lm = self._lm_loss(image_embeds, input_ids, attention_mask)
        return BLIPOutput(loss=loss_itc + loss_itm + loss_lm, loss_itc=loss_itc, loss_itm=loss_itm, loss_lm=loss_lm)

    # ------------------------------------------------------------------
    # ITC: 对比学习 + 动量蒸馏
    # ------------------------------------------------------------------
    def _itc_loss(self, images, input_ids, attention_mask, image_feat, text_feat):
        with torch.no_grad():
            self._momentum_update()
            image_feat_m = F.normalize(self.vision_proj_m(self.visual_m(images)[:, 0]), dim=-1)
            text_feat_m = F.normalize(self.text_proj_m(self.text_m(input_ids, attention_mask, mode="text")[:, 0]), dim=-1)
            # 当前 batch + 队列 一起作为候选: [E, B + Q]
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone()], dim=1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone()], dim=1)

            # 动量蒸馏: 用动量模型的预测作为软标签, 缓解网络图文对里的噪声 (有些"负样本"其实也匹配)
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
            hard_targets = torch.zeros_like(sim_i2t_m)
            hard_targets.fill_diagonal_(1)
            i2t_targets = self.alpha * F.softmax(sim_i2t_m, dim=1) + (1 - self.alpha) * hard_targets
            t2i_targets = self.alpha * F.softmax(sim_t2i_m, dim=1) + (1 - self.alpha) * hard_targets

        sim_i2t = image_feat @ text_feat_all / self.temp  # [B, B + Q]
        sim_t2i = text_feat @ image_feat_all / self.temp
        loss_i2t = -(F.log_softmax(sim_i2t, dim=1) * i2t_targets).sum(dim=1).mean()
        loss_t2i = -(F.log_softmax(sim_t2i, dim=1) * t2i_targets).sum(dim=1).mean()

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        B = image_feat.shape[0]
        return (loss_i2t + loss_t2i) / 2, sim_i2t[:, :B]

    # ------------------------------------------------------------------
    # ITM: 图像引导的文本编码器 + 难负样本二分类
    # ------------------------------------------------------------------
    def _itm_loss(self, image_embeds, input_ids, attention_mask, sim_i2t):
        B = input_ids.shape[0]
        enc_ids = input_ids.clone()
        enc_ids[:, 0] = self.enc_token_id

        neg_image_index, neg_text_index = sample_hard_negatives(sim_i2t)
        # 正样本 B 对 + (难负图, 文) B 对 + (图, 难负文) B 对
        images_all = torch.cat([image_embeds, image_embeds[neg_image_index], image_embeds], dim=0)
        ids_all = torch.cat([enc_ids, enc_ids, enc_ids[neg_text_index]], dim=0)
        mask_all = torch.cat([attention_mask, attention_mask, attention_mask[neg_text_index]], dim=0)

        hidden = self.text(ids_all, mask_all, mode="multimodal", image_embeds=images_all)
        logits = self.itm_head(hidden[:, 0])  # 用 [ENC] 位置的输出做二分类
        labels = torch.cat([torch.ones(B), torch.zeros(2 * B)]).long().to(logits.device)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # LM: 图像引导的文本解码器
    # ------------------------------------------------------------------
    def _lm_loss(self, image_embeds, input_ids, attention_mask):
        dec_ids = input_ids.clone()
        dec_ids[:, 0] = self.dec_token_id
        labels = dec_ids.masked_fill(attention_mask == 0, IGNORE_INDEX)
        hidden = self.text(dec_ids, attention_mask, mode="decoder", image_embeds=image_embeds)
        return causal_lm_loss(self.text.lm_head(hidden), labels, label_smoothing=0.1)

    # ------------------------------------------------------------------
    # 动量编码器 & 队列
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

    # ------------------------------------------------------------------
    # 推理: 图像描述生成 (贪心解码)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_caption(self, images: torch.Tensor, max_new_tokens: int = 20,
                         eos_token_id: Optional[int] = None) -> torch.Tensor:
        image_embeds = self.visual(images)
        ids = torch.full((images.shape[0], 1), self.dec_token_id, dtype=torch.long, device=images.device)
        for _ in range(max_new_tokens):
            hidden = self.text(ids, torch.ones_like(ids), mode="decoder", image_embeds=image_embeds)
            next_token = self.text.lm_head(hidden[:, -1]).argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
            if eos_token_id is not None and (ids == eos_token_id).any(dim=1).all():
                break
        return ids[:, 1:]
