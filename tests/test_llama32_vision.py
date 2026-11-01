import torch

from conftest import VOCAB_SIZE, has_grad
from mllm_algo.llama32_vision import (MllamaForConditionalGeneration, MllamaTextModel, MllamaVisionEncoder,
                                      build_cross_attention_mask, get_optimal_canvas, image_token_spans,
                                      preprocess_images)

TILE_SIZE, MAX_TILES = 16, 4
TOKENS_PER_TILE = (16 // 8) ** 2 + 1  # 4 个 patch + [CLS]
IMAGE_TOKEN_ID = VOCAB_SIZE  # 和官方一样, <|image|> 放在原词表之外
CROSS_LAYERS = (2, 5)


def build_model() -> MllamaForConditionalGeneration:
    vision = MllamaVisionEncoder(image_size=TILE_SIZE, patch_size=8, dim=32, num_heads=4, num_layers=4,
                                 num_global_layers=2, max_num_tiles=MAX_TILES, intermediate_layers=(1, 3))
    text = MllamaTextModel(VOCAB_SIZE, dim=48, num_layers=6, num_heads=4, num_kv_heads=2, ffn_dim=96,
                           cross_attention_layers=CROSS_LAYERS)
    return MllamaForConditionalGeneration(vision, text, IMAGE_TOKEN_ID)


def batch_inputs():
    """2 条样本, 每条两张不同尺寸的图: "<|image|> 文本 <|image|> 文本" """
    samples = [preprocess_images([torch.rand(3, 20, 40), torch.rand(3, 30, 30)], TILE_SIZE, MAX_TILES)
               for _ in range(2)]
    pixel_values, aspect_ratio_ids, aspect_ratio_mask = (torch.stack(x) for x in zip(*samples))
    input_ids = torch.randint(5, VOCAB_SIZE, (2, 10))
    input_ids[:, [0, 5]] = IMAGE_TOKEN_ID
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 8:] = 0
    return dict(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids, aspect_ratio_mask=aspect_ratio_mask)


def test_optimal_canvas():
    assert get_optimal_canvas(100, 200, tile_size=32, max_num_tiles=4) == (2, 1)  # 宽图: 2 列 1 行
    assert get_optimal_canvas(20, 20, tile_size=32, max_num_tiles=4) == (1, 1)  # 小图: 1 个 tile 就够
    assert get_optimal_canvas(200, 200, tile_size=32, max_num_tiles=4) == (2, 2)


def test_preprocess_images():
    pixel_values, ratio_ids, ratio_mask = preprocess_images([torch.rand(3, 20, 40)], TILE_SIZE, MAX_TILES)
    assert pixel_values.shape == (1, MAX_TILES, 3, TILE_SIZE, TILE_SIZE)
    assert ratio_mask.tolist() == [[1, 1, 0, 0]]  # 2 个真实 tile + 2 个补齐
    assert ratio_ids.item() > 0


def test_image_token_spans():
    I = IMAGE_TOKEN_ID
    assert image_token_spans(torch.tensor([I, 7, 7, I, 7]), I) == [(0, 3), (3, 5)]  # 只看最近的图
    assert image_token_spans(torch.tensor([I, I, 7, 7]), I) == [(0, 4), (1, 4)]  # 连续的图一起看
    assert image_token_spans(torch.tensor([7, 7]), I) == []


def test_cross_attention_mask_hides_padding_tiles():
    input_ids = torch.tensor([[IMAGE_TOKEN_ID, 7, 7]])
    aspect_ratio_mask = torch.tensor([[[1, 1, 0, 0]]])
    mask = build_cross_attention_mask(input_ids, IMAGE_TOKEN_ID, aspect_ratio_mask, tokens_per_tile=2)
    assert mask.shape == (1, 3, 4 * 2)
    assert mask[0, 1].tolist() == [True] * 4 + [False] * 4


def test_forward_backward_trains_only_new_parts():
    model = build_model()
    out = model(**batch_inputs())
    assert torch.isfinite(out.loss)
    out.loss.backward()

    lm = model.language_model
    assert has_grad(model.vision_model) and has_grad(model.multi_modal_projector)
    assert all(has_grad(lm.layers[i]) for i in CROSS_LAYERS)
    assert not has_grad(lm.layers[0]) and not has_grad(lm.embed_tokens) and not has_grad(lm.lm_head)


def test_gates_start_closed_and_text_only_matches_llama():
    """cross-attention 门控初始为 0: 有图和无图的输出一致, 都等于原始 Llama。"""
    model = build_model().eval()
    inputs = batch_inputs()
    with_image = model(**inputs).logits
    text_only = model(inputs["input_ids"], inputs["attention_mask"]).logits
    assert torch.allclose(with_image, text_only, atol=1e-5)


def test_generate():
    model = build_model()
    assert model.generate(**batch_inputs(), max_new_tokens=3).shape == (2, 3)
