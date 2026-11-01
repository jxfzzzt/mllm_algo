import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def contrastive_loss(logits_per_image: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """对称的 InfoNCE loss。logits_per_image: [B, B], 对角线是匹配的图文对。"""
    targets = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    loss_i2t = F.cross_entropy(logits_per_image, targets, label_smoothing=label_smoothing)
    loss_t2i = F.cross_entropy(logits_per_image.t(), targets, label_smoothing=label_smoothing)
    return (loss_i2t + loss_t2i) / 2


@torch.no_grad()
def sample_hard_negatives(sim_i2t: torch.Tensor):
    """ITM 的难负样本挖掘 (ALBEF / BLIP / BLIP-2 通用)。

    sim_i2t: [B, B] 图文相似度。对每个文本按相似度采样一张"像但不对"的图, 对每张图采样一个难负文本。
    return: (neg_image_index [B], neg_text_index [B])
    """
    weights_t2i = F.softmax(sim_i2t.t().float(), dim=1).fill_diagonal_(0)  # 正样本概率置 0
    weights_i2t = F.softmax(sim_i2t.float(), dim=1).fill_diagonal_(0)
    neg_image_index = torch.multinomial(weights_t2i, 1).squeeze(1)
    neg_text_index = torch.multinomial(weights_i2t, 1).squeeze(1)
    return neg_image_index, neg_text_index


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """自回归语言模型 loss: 用第 t 个位置的输出预测第 t+1 个 token, labels 中的 -100 不计入 loss。"""
    shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX, label_smoothing=label_smoothing)
