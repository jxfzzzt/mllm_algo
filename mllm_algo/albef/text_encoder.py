"""ALBEF 的文本侧: 把 BERT-base 的 12 层拆成两半。

    前 6 层  文本编码器   纯文本 self-attention            -> [CLS] 用于 ITC
    后 6 层  融合编码器   self-attention + 对图像的 cross-attention -> 用于 ITM / MLM

"Align before Fuse" 的含义: 先用 ITC 把单模态的图像特征和文本特征对齐 (前 6 层的输出),
再送进融合编码器做跨模态交互, 融合时更容易学到图文对应关系。
"""
import torch
import torch.nn as nn

from ..common import TransformerBlock, padding_mask


class ALBEFTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 768, num_text_layers: int = 6, num_fusion_layers: int = 6,
                 num_heads: int = 12, image_dim: int = 768, max_len: int = 512):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.embed_norm = nn.LayerNorm(dim)

        self.text_layers = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(num_text_layers)])
        self.text_norm = nn.LayerNorm(dim)
        self.fusion_layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, context_dim=image_dim) for _ in range(num_fusion_layers)
        ])
        self.fusion_norm = nn.LayerNorm(dim)

        # BERT 的 MLM 预测头: Linear -> GELU -> LayerNorm -> 词表
        self.mlm_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, vocab_size))

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """前 6 层, 不看图像。return: [B, L, D]"""
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.embed_norm(self.token_embed(input_ids) + self.pos_embed(positions))
        mask = padding_mask(attention_mask)
        for layer in self.text_layers:
            x = layer(x, mask=mask)
        return self.text_norm(x)

    def fuse(self, text_embeds: torch.Tensor, attention_mask: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        """后 6 层, 以文本编码器的输出为输入, 每层通过 cross-attention 看图像。return: [B, L, D]"""
        x = text_embeds
        mask = padding_mask(attention_mask)
        for layer in self.fusion_layers:
            x = layer(x, mask=mask, context=image_embeds)
        return self.fusion_norm(x)

    def mlm_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                   image_embeds: torch.Tensor) -> torch.Tensor:
        """完整的 12 层 + MLM 头, 借助图像预测被 mask 的词。return: [B, L, vocab]"""
        text_embeds = self.encode_text(input_ids, attention_mask)
        return self.mlm_head(self.fuse(text_embeds, attention_mask, image_embeds))
