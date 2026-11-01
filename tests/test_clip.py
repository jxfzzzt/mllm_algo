import torch

from conftest import VOCAB_SIZE, has_grad, tiny_vit
from mllm_algo.clip import CLIP, CLIPTextEncoder


def build_clip() -> CLIP:
    text_encoder = CLIPTextEncoder(VOCAB_SIZE, dim=32, depth=2, num_heads=4, max_len=16)
    return CLIP(tiny_vit(), text_encoder, embed_dim=16)


def test_forward_backward(images, text):
    model = build_clip()
    out = model(images, *text)
    assert out.logits.shape == (4, 4)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert has_grad(model.visual) and has_grad(model.text) and model.logit_scale.grad is not None


def test_zero_shot_classify(images, text):
    model = build_clip()
    probs = model.zero_shot_classify(images, *text)
    assert probs.shape == (4, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4))
