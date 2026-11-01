import torch
import torch.nn as nn

from ..common import TransformerBlock, causal_padding_mask


class CLIPTextEncoder(nn.Module):
    """CLIP 的文本编码器: 带因果 mask 的 Transformer, 取 [EOS] 位置 (最后一个有效 token) 的输出作为句子表示。"""

    def __init__(self, vocab_size: int, dim: int = 512, depth: int = 12, num_heads: int = 8, max_len: int = 77):
        super().__init__()
        self.dim = dim
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.01)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids / attention_mask: [B, L], 右 padding。return: [B, D]"""
        L = input_ids.shape[1]
        x = self.token_embed(input_ids) + self.pos_embed[:, :L]
        mask = causal_padding_mask(attention_mask)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)

        # 因果 mask 下只有最后一个 token 看到了整句话, 所以用 [EOS] 位置做句子表示
        eos_index = attention_mask.sum(dim=1) - 1
        return x[torch.arange(x.shape[0], device=x.device), eos_index]
