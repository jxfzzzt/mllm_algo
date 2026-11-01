"""BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs (Li et al., 2023)

    冻结的 ViT ──> Q-Former (32 个 query) ──> Linear ──> 冻结的 LLM

两阶段训练, 图像编码器和 LLM 始终冻结, 只训练 Q-Former (~188M) 和投影层:
    Stage 1 表示学习 (Blip2QFormerPretrain):  只接 ViT, 用 ITC + ITM + ITG 训练 Q-Former,
                                             让 query 学会提取和文本相关的视觉特征
    Stage 2 生成学习 (Blip2ForCausalLM):      Q-Former 输出投影成 LLM 的 soft visual prompt,
                                             拼在文本前面, 只用语言模型 loss 训练
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import (IGNORE_INDEX, ModelOutput, VisionTransformer, causal_lm_loss, contrastive_loss, freeze,
                      padding_mask, sample_hard_negatives)
from .qformer import QFormer, build_qformer_mask

VISION_LAYER = -2  # BLIP-2 去掉 ViT 最后一层, 使用倒数第二层的输出


@dataclass
class Blip2Stage1Output:
    loss: torch.Tensor
    loss_itc: torch.Tensor
    loss_itm: torch.Tensor
    loss_itg: torch.Tensor


class Blip2QFormerPretrain(nn.Module):
    """Stage 1: 视觉-语言表示学习。"""

    def __init__(self, vision_encoder: VisionTransformer, qformer: QFormer, dec_token_id: int, embed_dim: int = 256):
        super().__init__()
        self.visual = freeze(vision_encoder)
        self.ln_vision = nn.LayerNorm(vision_encoder.dim)
        self.qformer = qformer
        self.vision_proj = nn.Linear(qformer.dim, embed_dim)
        self.text_proj = nn.Linear(qformer.dim, embed_dim)
        self.itm_head = nn.Linear(qformer.dim, 2)
        self.temp = nn.Parameter(torch.tensor(0.07))
        self.dec_token_id = dec_token_id

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_embeds = self.visual(images, layer=VISION_LAYER)
        return self.ln_vision(image_embeds)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Blip2Stage1Output:
        """input_ids 第一个 token 是 [CLS], 右 padding。"""
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        image_embeds = self.encode_image(images)

        loss_itc, sim_i2t = self._itc_loss(image_embeds, input_ids, attention_mask)
        loss_itm = self._itm_loss(image_embeds, input_ids, attention_mask, sim_i2t)
        loss_itg = self._itg_loss(image_embeds, input_ids, attention_mask)
        return Blip2Stage1Output(loss=loss_itc + loss_itm + loss_itg,
                                 loss_itc=loss_itc, loss_itm=loss_itm, loss_itg=loss_itg)

    def _itc_loss(self, image_embeds, input_ids, attention_mask):
        """ITC: 单模态 mask, query 和 text 分别前向。"""
        query_out, _ = self.qformer(image_embeds=image_embeds)
        image_feat = F.normalize(self.vision_proj(query_out), dim=-1)  # [B, N_q, E]
        _, text_out = self.qformer(input_ids=input_ids, mask=padding_mask(attention_mask))
        text_feat = F.normalize(self.text_proj(text_out[:, 0]), dim=-1)  # [CLS], [B, E]

        # 每个 query 分别和文本算相似度, 取最大值作为图文相似度 (不同 query 关注图像的不同方面)
        sim_query2text = torch.einsum("iqe,je->ijq", image_feat, text_feat)  # [B_img, B_text, N_q]
        sim_i2t = sim_query2text.max(dim=-1).values / self.temp  # [B, B]
        return contrastive_loss(sim_i2t, label_smoothing=0.1), sim_i2t

    def _itm_loss(self, image_embeds, input_ids, attention_mask, sim_i2t):
        """ITM: 双向 mask + 难负样本, 每个 query 的输出都做二分类再取平均。"""
        B = input_ids.shape[0]
        neg_image_index, neg_text_index = sample_hard_negatives(sim_i2t)
        images_all = torch.cat([image_embeds, image_embeds[neg_image_index], image_embeds], dim=0)
        ids_all = torch.cat([input_ids, input_ids, input_ids[neg_text_index]], dim=0)
        mask_all = torch.cat([attention_mask, attention_mask, attention_mask[neg_text_index]], dim=0)

        mask = build_qformer_mask(self.qformer.num_query, mask_all, mode="bidirectional")
        query_out, _ = self.qformer(images_all, ids_all, mask)
        logits = self.itm_head(query_out).mean(dim=1)  # [3B, 2]
        labels = torch.cat([torch.ones(B), torch.zeros(2 * B)]).long().to(logits.device)
        return F.cross_entropy(logits, labels)

    def _itg_loss(self, image_embeds, input_ids, attention_mask):
        """ITG: 多模态因果 mask。text 看不到图像, 只能通过 query 获取视觉信息, 迫使 query 提取完整的视觉信息。"""
        dec_ids = input_ids.clone()
        dec_ids[:, 0] = self.dec_token_id  # [CLS] -> [DEC]
        labels = dec_ids.masked_fill(attention_mask == 0, IGNORE_INDEX)

        mask = build_qformer_mask(self.qformer.num_query, attention_mask, mode="causal")
        _, text_out = self.qformer(image_embeds, dec_ids, mask)
        return causal_lm_loss(self.qformer.lm_head(text_out), labels, label_smoothing=0.1)


class Blip2ForCausalLM(nn.Module):
    """Stage 2: 视觉到语言的生成学习, 以 decoder-only LLM (OPT / LLaMA) 为例。

    llm 可以是 common.CausalLM, 也可以直接换成 HuggingFace 的 AutoModelForCausalLM。
    """

    def __init__(self, vision_encoder: VisionTransformer, ln_vision: nn.LayerNorm, qformer: QFormer, llm: nn.Module):
        super().__init__()
        self.visual = freeze(vision_encoder)
        self.ln_vision = ln_vision
        self.qformer = qformer
        self.llm = freeze(llm)
        # 唯一新增的模块: 把 Q-Former 输出投影到 LLM 的词向量空间
        self.llm_proj = nn.Linear(qformer.dim, llm.config.hidden_size)

    @classmethod
    def from_stage1(cls, stage1: Blip2QFormerPretrain, llm: nn.Module) -> "Blip2ForCausalLM":
        return cls(stage1.visual, stage1.ln_vision, stage1.qformer, llm)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """return: [B, N_q, D_llm] 的 soft visual prompt"""
        with torch.no_grad():
            image_embeds = self.visual(images, layer=VISION_LAYER)
        query_out, _ = self.qformer(image_embeds=self.ln_vision(image_embeds))  # stage 2 只把 query 送进 Q-Former
        return self.llm_proj(query_out)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutput:
        """input_ids: LLM tokenizer 编码的文本 (图像描述, 或 prompt + 答案), 右 padding。"""
        visual_embeds = self.encode_image(images)
        B, num_query, _ = visual_embeds.shape

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        text_labels = input_ids.masked_fill(attention_mask == 0, IGNORE_INDEX)

        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # [视觉 token ; 文本 token]
        attention_mask = torch.cat([attention_mask.new_ones(B, num_query), attention_mask], dim=1)
        labels = torch.cat([text_labels.new_full((B, num_query), IGNORE_INDEX), text_labels], dim=1)  # 视觉 token 不算 loss
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor, **generate_kwargs):
        """prompt 例如 "Question: {} Answer:", 批量时使用左 padding。"""
        visual_embeds = self.encode_image(images)
        text_embeds = self.llm.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        attention_mask = torch.cat([prompt_mask.new_ones(visual_embeds.shape[:2]), prompt_mask], dim=1)
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs)
