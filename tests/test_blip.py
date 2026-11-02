import torch

from conftest import VOCAB_SIZE, has_grad, tiny_vit
from mllm_algo.blip import MED, BLIPPretrain

ENC_TOKEN_ID, DEC_TOKEN_ID, CLS_TOKEN_ID = 1, 2, 3


def build_blip() -> BLIPPretrain:
    med = MED(VOCAB_SIZE, dim=32, depth=2, num_heads=4, image_dim=32, max_len=32)
    return BLIPPretrain(tiny_vit(), med, ENC_TOKEN_ID, DEC_TOKEN_ID, embed_dim=16, queue_size=8)


def with_cls(text):
    """BLIPPretrain 约定 input_ids 第 0 个 token 是 [CLS] (ITM / LM 会把它替换成 [ENC] / [DEC])。"""
    input_ids, attention_mask = text
    input_ids = input_ids.clone()
    input_ids[:, 0] = CLS_TOKEN_ID
    return input_ids, attention_mask


def test_forward_backward(images, text):
    model = build_blip()
    out = model(images, *with_cls(text))
    for loss in (out.loss_itc, out.loss_itm, out.loss_lm):
        assert torch.isfinite(loss)
    out.loss.backward()
    assert has_grad(model.visual) and has_grad(model.text)
    assert not has_grad(model.visual_m) and not has_grad(model.text_m)  # 动量编码器不参与反传


def test_queue_is_updated(images, text):
    model = build_blip()
    queue_before = model.image_queue.clone()
    model(images, *with_cls(text))
    assert int(model.queue_ptr) == 4
    assert not torch.equal(model.image_queue[:, :4], queue_before[:, :4])
    assert torch.equal(model.image_queue[:, 4:], queue_before[:, 4:])


def test_decoder_shares_all_but_self_attention():
    med = MED(VOCAB_SIZE, dim=32, depth=1, num_heads=4, image_dim=32)
    layer = med.layers[0]
    assert layer.self_attn is not layer.self_attn_dec


def test_generate_caption(images):
    model = build_blip()
    caption = model.generate_caption(images, max_new_tokens=5)
    assert caption.shape == (4, 5)


def test_generate_caption_pads_after_eos(images):
    model = build_blip()
    reference = model.generate_caption(images, max_new_tokens=5)
    eos = int(reference[0, 1])  # 让第 0 条样本最晚在第 2 步结束
    caption = model.generate_caption(images, max_new_tokens=5, eos_token_id=eos, pad_token_id=0)
    for row, ref in zip(caption, reference):
        hits = (row == eos).nonzero()
        if len(hits) == 0:
            assert torch.equal(row, ref[:len(row)])
            continue
        end = int(hits[0]) + 1
        assert torch.equal(row[:end], ref[:end])  # EOS 及之前与不设 EOS 时一致
        assert (row[end:] == 0).all()  # EOS 之后全是 pad
