from typing import Sequence, Tuple

import torch

from ..common import IGNORE_INDEX


def mask_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, vocab_size: int, mask_token_id: int,
                special_token_ids: Sequence[int] = (), mask_prob: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """BERT 式的随机 mask: 选 15% 的 token 作为预测目标, 其中

        80% 替换成 [MASK]
        10% 替换成随机词
        10% 保持不变      (让模型不能只在看到 [MASK] 时才认真预测)

    padding 和特殊 token ([CLS] / [SEP] 等) 不会被选中。
    return: (mask 后的 input_ids, labels), 未被选中的位置 label 为 -100
    """
    device = input_ids.device
    prob = torch.full(input_ids.shape, mask_prob, device=device)
    prob.masked_fill_(attention_mask == 0, 0.0)
    for token_id in special_token_ids:
        prob.masked_fill_(input_ids == token_id, 0.0)
    selected = torch.bernoulli(prob).bool()

    labels = input_ids.masked_fill(~selected, IGNORE_INDEX)
    masked_ids = input_ids.clone()

    to_mask = selected & torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool()
    masked_ids[to_mask] = mask_token_id

    # 剩下 20% 中的一半 (即总体的 10%) 换成随机词
    to_random = selected & ~to_mask & torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool()
    masked_ids[to_random] = torch.randint(vocab_size, input_ids.shape, device=device)[to_random]
    return masked_ids, labels
