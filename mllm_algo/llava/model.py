"""LLaVA: Visual Instruction Tuning (Liu et al., 2023)  /  LLaVA-1.5 (Liu et al., 2023)

    image ──冻结的 CLIP ViT (倒数第二层, 去掉 [CLS])──> projector ──> N_patch 个视觉 token ─┐
                                                                                         ↓ 替换 <image> 占位符
    "USER: <image>\\n问题 ASSISTANT: 回答" ──────────────────────────────> LLM (Vicuna / LLaMA)

结构极简: 视觉 patch 特征经过投影后直接当作 token 拼进 LLM 的输入序列。
两阶段训练:
    Stage 1 特征对齐预训练  只训练 projector, 数据是图文对 (CC3M / LCS-558K), loss 只算描述部分
    Stage 2 视觉指令微调    训练 projector + LLM, 数据是 GPT-4 生成的多轮对话, loss 只算 ASSISTANT 的回答
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..common import IGNORE_INDEX, VisionTransformer, freeze
from .projector import build_projector

IMAGE_TOKEN_INDEX = -200  # input_ids 中 <image> 占位符的 id (负数, 不在词表里), 和官方实现一致


class LlavaForCausalLM(nn.Module):
    def __init__(self, vision_tower: VisionTransformer, llm: nn.Module, projector_type: str = "mlp2x_gelu",
                 vision_select_layer: int = -2):
        super().__init__()
        self.vision_tower = freeze(vision_tower)  # 两个阶段都冻结
        self.llm = llm
        self.vision_select_layer = vision_select_layer  # 倒数第二层的特征比最后一层更通用 (最后一层偏向对比学习目标)
        self.projector = build_projector(projector_type, vision_tower.dim, llm.config.hidden_size)
        self.set_train_stage(1)

    def set_train_stage(self, stage: int):
        """stage 1: 只训练 projector; stage 2: 训练 projector + LLM。"""
        assert stage in (1, 2)
        for p in self.llm.parameters():
            p.requires_grad = stage == 2

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """return: [B, N_patch, D_llm]"""
        with torch.no_grad():
            features = self.vision_tower(images, layer=self.vision_select_layer)[:, 1:]  # 去掉 [CLS], 只用 patch
        return self.projector(features)

    def prepare_inputs_embeds(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                              image_features: torch.Tensor, labels: Optional[torch.Tensor] = None,
                              padding_side: str = "right") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """LLaVA 的核心步骤: 把每条样本里的 <image> 占位符替换成该图像的 N_patch 个视觉 token, 再重新 padding。

        替换后各样本长度可能不同 (例如有的样本没有图), 所以逐条拼接后统一 padding。
        训练用右 padding, 生成用左 padding。
        return: (inputs_embeds [B, L', D], attention_mask [B, L'], labels [B, L'])
        """
        embed_tokens = self.llm.get_input_embeddings()
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        new_embeds, new_labels = [], []
        for ids, mask, lab, image in zip(input_ids, attention_mask, labels, image_features):
            ids, lab = ids[mask.bool()], lab[mask.bool()]  # 先去掉原来的 padding
            text_embeds = embed_tokens(ids.clamp(min=0))  # 占位符 id 为负, clamp 后取到的向量会被丢掉

            image_pos = (ids == IMAGE_TOKEN_INDEX).nonzero().flatten()
            assert len(image_pos) <= 1, "这里只实现了每条样本最多一张图"
            if len(image_pos) == 0:
                new_embeds.append(text_embeds)
                new_labels.append(lab)
                continue

            p = image_pos.item()
            image_labels = lab.new_full((image.shape[0],), IGNORE_INDEX)  # 视觉 token 不算 loss
            new_embeds.append(torch.cat([text_embeds[:p], image, text_embeds[p + 1:]]))
            new_labels.append(torch.cat([lab[:p], image_labels, lab[p + 1:]]))

        return self._pad(new_embeds, new_labels, padding_side)

    # 统一做padding的方法
    @staticmethod
    def _pad(embeds_list, labels_list, padding_side: str):
        B, max_len, D = len(embeds_list), max(e.shape[0] for e in embeds_list), embeds_list[0].shape[-1]
        ref = embeds_list[0]
        inputs_embeds = ref.new_zeros(B, max_len, D)
        labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long, device=ref.device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=ref.device)

        for i, (emb, lab) in enumerate(zip(embeds_list, labels_list)):
            L = emb.shape[0]
            span = slice(0, L) if padding_side == "right" else slice(max_len - L, max_len)
            inputs_embeds[i, span] = emb
            labels[i, span] = lab
            attention_mask[i, span] = 1
        return inputs_embeds, attention_mask, labels

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor):
        """labels 由数据预处理决定: 和 input_ids 相同, 但不需要算 loss 的位置 (system / USER 部分) 设为 -100。"""
        image_features = self.encode_images(images)
        inputs_embeds, attention_mask, labels = self.prepare_inputs_embeds(
            input_ids, attention_mask, image_features, labels, padding_side="right")
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 **generate_kwargs) -> torch.Tensor:
        image_features = self.encode_images(images)
        inputs_embeds, attention_mask, _ = self.prepare_inputs_embeds(
            input_ids, attention_mask, image_features, padding_side="left")
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs)
