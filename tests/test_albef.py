import torch

from conftest import VOCAB_SIZE, has_grad, tiny_vit
from mllm_algo.albef import ALBEFPretrain, ALBEFTextEncoder, mask_tokens
from mllm_algo.common import IGNORE_INDEX

CLS_TOKEN_ID, MASK_TOKEN_ID = 1, 4


def build_albef() -> ALBEFPretrain:
    text = ALBEFTextEncoder(VOCAB_SIZE, dim=32, num_text_layers=2, num_fusion_layers=2, num_heads=4,
                            image_dim=32, max_len=32)
    return ALBEFPretrain(tiny_vit(), text, MASK_TOKEN_ID, special_token_ids=(CLS_TOKEN_ID,),
                         embed_dim=16, queue_size=8, mlm_prob=0.3)


def with_cls(text):
    input_ids, attention_mask = text
    input_ids = input_ids.clone()
    input_ids[:, 0] = CLS_TOKEN_ID
    return input_ids, attention_mask


def test_forward_backward(images, text):
    model = build_albef()
    out = model(images, *with_cls(text))
    for loss in (out.loss_itc, out.loss_itm, out.loss_mlm):
        assert torch.isfinite(loss) and loss > 0
    out.loss.backward()
    assert has_grad(model.visual) and has_grad(model.text.text_layers) and has_grad(model.text.fusion_layers)
    assert not has_grad(model.visual_m) and not has_grad(model.text_m)  # 动量模型不参与反传


def test_queue_is_updated(images, text):
    model = build_albef()
    queue_before = model.text_queue.clone()
    model(images, *with_cls(text))
    assert int(model.queue_ptr) == 4
    assert not torch.equal(model.text_queue[:, :4], queue_before[:, :4])
    assert torch.equal(model.text_queue[:, 4:], queue_before[:, 4:])


def test_text_layers_do_not_see_image():
    """前半部分是纯文本编码器 (没有 cross-attention), 后半部分是融合编码器。"""
    text = ALBEFTextEncoder(VOCAB_SIZE, dim=32, num_text_layers=2, num_fusion_layers=3, num_heads=4, image_dim=32)
    assert not any(layer.has_cross_attn for layer in text.text_layers)
    assert all(layer.has_cross_attn for layer in text.fusion_layers)


def test_mask_tokens():
    input_ids = torch.randint(5, VOCAB_SIZE, (64, 32))
    input_ids[:, 0] = CLS_TOKEN_ID
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, 24:] = 0
    masked_ids, labels = mask_tokens(input_ids, attention_mask, VOCAB_SIZE, MASK_TOKEN_ID,
                                     special_token_ids=(CLS_TOKEN_ID,))
    selected = labels != IGNORE_INDEX

    assert not selected[:, 0].any() and not selected[:, 24:].any()  # [CLS] 和 padding 不会被选中
    assert torch.equal(labels[selected], input_ids[selected])  # label 是原来的词
    assert torch.equal(masked_ids[~selected], input_ids[~selected])  # 没选中的位置不变
    assert 0.10 < selected[:, 1:24].float().mean() < 0.20  # 约 15%
    assert (masked_ids[selected] == MASK_TOKEN_ID).float().mean() > 0.6  # 大部分换成 [MASK]
