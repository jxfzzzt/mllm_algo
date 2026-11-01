"""一个最小的 decoder-only 语言模型 (GPT 风格), 作为 BLIP-2 / Flamingo / LLaVA 里"冻结的 LLM"。

接口刻意和 HuggingFace 的 CausalLM 保持一致 (get_input_embeddings / inputs_embeds / labels / generate),
因此 BLIP-2 和 LLaVA 的代码可以直接换成 OPT、LLaMA、Vicuna 等真实模型。
"""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

from .attention import TransformerBlock, causal_padding_mask
from .losses import causal_lm_loss


@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class CausalLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 768, depth: int = 12, num_heads: int = 12,
                 max_len: int = 2048):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, hidden_size=dim)
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def embed(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """词向量加上位置编码。位置由 attention_mask 累加得到, 这样左 padding 也能拿到正确的位置。"""
        position_ids = (attention_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        return inputs_embeds + self.pos_embed(position_ids)

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> ModelOutput:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

        x = self.embed(inputs_embeds, attention_mask)
        mask = causal_padding_mask(attention_mask)
        for layer in self.layers:
            x = layer(x, mask=mask)
        logits = self.lm_head(self.norm(x))

        loss = causal_lm_loss(logits, labels) if labels is not None else None
        return ModelOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.Tensor] = None,
                 inputs_embeds: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 20, eos_token_id: Optional[int] = None) -> torch.Tensor:
        """贪心解码, 只返回新生成的 token。为了可读性没有实现 KV cache。批量 prompt 请使用左 padding。"""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        B = inputs_embeds.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

        generated = []
        finished = torch.zeros(B, dtype=torch.bool, device=inputs_embeds.device)
        for _ in range(max_new_tokens):
            logits = self(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
            next_token = logits[:, -1].argmax(dim=-1)  # [B]
            generated.append(next_token)

            inputs_embeds = torch.cat([inputs_embeds, self.embed_tokens(next_token)[:, None]], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(B, 1)], dim=1)
            if eos_token_id is not None:
                finished |= next_token == eos_token_id
                if finished.all():
                    break
        return torch.stack(generated, dim=1)
