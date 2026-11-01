import torch

from conftest import has_grad, tiny_llm, tiny_vit
from mllm_algo.common import IGNORE_INDEX as IGNORE
from mllm_algo.llava import IMAGE_TOKEN_INDEX, LlavaForCausalLM

NUM_PATCHES = 16


def build_llava() -> LlavaForCausalLM:
    return LlavaForCausalLM(tiny_vit(), tiny_llm(), projector_type="mlp2x_gelu")


def conversation(batch_size=4, length=10):
    """模拟 "USER: <image> 问题 ASSISTANT: 回答", 只有回答部分 (第 6 个 token 之后) 计算 loss。第 0 条样本较短。"""
    input_ids = torch.randint(5, 100, (batch_size, length))
    input_ids[:, 1] = IMAGE_TOKEN_INDEX
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, 8:] = 0
    labels = input_ids.clone()
    labels[:, :6] = IGNORE
    labels[attention_mask == 0] = IGNORE
    return input_ids, attention_mask, labels


def test_prepare_inputs_embeds_splices_image_tokens():
    model = build_llava()
    input_ids, attention_mask, labels = conversation()
    image_features = torch.randn(4, NUM_PATCHES, 48)
    embeds, new_mask, new_labels = model.prepare_inputs_embeds(input_ids, attention_mask, image_features, labels)

    # <image> (1 个 token) 被替换成 16 个视觉 token
    assert embeds.shape == (4, 10 - 1 + NUM_PATCHES, 48)
    assert torch.equal(embeds[1, 1:1 + NUM_PATCHES], image_features[1])
    assert new_mask.sum(dim=1).tolist() == [8 - 1 + NUM_PATCHES] + [10 - 1 + NUM_PATCHES] * 3
    assert (new_labels[:, 1:1 + NUM_PATCHES] == IGNORE).all()  # 视觉 token 不算 loss
    assert (new_labels != IGNORE).sum() == (labels != IGNORE).sum()  # 回答部分的 label 原样保留


def test_left_padding_for_generation():
    model = build_llava()
    input_ids, attention_mask, _ = conversation()
    _, new_mask, _ = model.prepare_inputs_embeds(input_ids, attention_mask, torch.randn(4, NUM_PATCHES, 48),
                                                 padding_side="left")
    assert new_mask[0, :2].sum() == 0 and new_mask[0, -1] == 1


def test_stage1_trains_only_projector(images):
    model = build_llava()
    model(images, *conversation()).loss.backward()
    assert has_grad(model.projector)
    assert not has_grad(model.llm) and not has_grad(model.vision_tower)


def test_stage2_trains_projector_and_llm(images):
    model = build_llava()
    model.set_train_stage(2)
    model(images, *conversation()).loss.backward()
    assert has_grad(model.projector) and has_grad(model.llm)
    assert not has_grad(model.vision_tower)


def test_generate(images):
    model = build_llava()
    input_ids, attention_mask, _ = conversation()
    assert model.generate(images, input_ids, attention_mask, max_new_tokens=4).shape == (4, 4)
