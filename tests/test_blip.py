import torch

from conftest import VOCAB_SIZE, has_grad, tiny_vit
from mllm_algo.blip import MED, BLIPPretrain

ENC_TOKEN_ID, DEC_TOKEN_ID = 1, 2


def build_blip() -> BLIPPretrain:
    med = MED(VOCAB_SIZE, dim=32, depth=2, num_heads=4, image_dim=32, max_len=32)
    return BLIPPretrain(tiny_vit(), med, ENC_TOKEN_ID, DEC_TOKEN_ID, embed_dim=16, queue_size=8)


def test_forward_backward(images, text):
    model = build_blip()
    out = model(images, *text)
    for loss in (out.loss_itc, out.loss_itm, out.loss_lm):
        assert torch.isfinite(loss)
    out.loss.backward()
    assert has_grad(model.visual) and has_grad(model.text)
    assert not has_grad(model.visual_m) and not has_grad(model.text_m)  # 动量编码器不参与反传


def test_queue_is_updated(images, text):
    model = build_blip()
    queue_before = model.image_queue.clone()
    model(images, *text)
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
