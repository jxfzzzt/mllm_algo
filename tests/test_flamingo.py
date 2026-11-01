import torch

from conftest import has_grad, tiny_llm, tiny_vit
from mllm_algo.flamingo import Flamingo, PerceiverResampler, build_media_mask

MEDIA_TOKEN_ID = 3
NUM_LATENTS = 4


def build_flamingo() -> Flamingo:
    return Flamingo(tiny_vit(), tiny_llm(), MEDIA_TOKEN_ID, cross_attn_every=2,
                    perceiver_depth=2, num_latents=NUM_LATENTS, num_heads=4)


def interleaved_text(batch_size=4, length=12):
    """两张图交错: "<image> 文本 <image> 文本" """
    input_ids = torch.randint(5, 100, (batch_size, length))
    input_ids[:, 0] = MEDIA_TOKEN_ID
    input_ids[:, 6] = MEDIA_TOKEN_ID
    return input_ids, torch.ones_like(input_ids)


def test_media_mask_attends_only_latest_image():
    input_ids = torch.tensor([[7, MEDIA_TOKEN_ID, 7, MEDIA_TOKEN_ID, 7]])
    mask = build_media_mask(input_ids, MEDIA_TOKEN_ID, num_images=2, num_latents=2)[0]
    assert not mask[0].any()  # 第一张图之前的 token 看不到任何图
    assert mask[2].tolist() == [True, True, False, False]
    assert mask[4].tolist() == [False, False, True, True]


def test_perceiver_output_shape():
    resampler = PerceiverResampler(dim=32, depth=2, num_heads=4, num_latents=NUM_LATENTS)
    assert resampler(torch.randn(2, 3, 10, 32)).shape == (2, NUM_LATENTS, 32)


def test_gates_start_closed():
    """门控初始化为 0, 训练开始时模型输出和原始 LM 完全一致。"""
    model = build_flamingo().eval()
    input_ids, attention_mask = interleaved_text()
    images = torch.randn(4, 2, 3, 32, 32)
    flamingo_logits = model(images, input_ids, attention_mask).logits
    lm_logits = model.lang_model(input_ids, attention_mask=attention_mask).logits
    assert torch.allclose(flamingo_logits, lm_logits, atol=1e-5)


def test_forward_backward_and_generate():
    model = build_flamingo()
    input_ids, attention_mask = interleaved_text()
    images = torch.randn(4, 2, 3, 32, 32)  # 每条样本两张图
    out = model(images, input_ids, attention_mask)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert has_grad(model.perceiver) and has_grad(model.xattn_layers)
    assert not has_grad(model.lang_model) and not has_grad(model.visual)
    assert len(model.xattn_layers) == 2  # 4 层 LM, 每 2 层插一个

    assert model.generate(images, input_ids, attention_mask, max_new_tokens=3).shape == (4, 3)
