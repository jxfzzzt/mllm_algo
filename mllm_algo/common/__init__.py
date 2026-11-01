from .attention import (FeedForward, MultiHeadAttention, TransformerBlock, causal_mask,
                        causal_padding_mask, padding_mask)
from .language_model import CausalLM, ModelOutput
from .losses import IGNORE_INDEX, causal_lm_loss, contrastive_loss, sample_hard_negatives
from .vision import VisionTransformer


def freeze(module):
    """冻结模块参数, 并切到 eval 模式。"""
    for p in module.parameters():
        p.requires_grad = False
    return module.eval()
