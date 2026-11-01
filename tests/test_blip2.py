import torch

from conftest import VOCAB_SIZE, has_grad, tiny_llm, tiny_vit
from mllm_algo.blip2 import Blip2ForCausalLM, Blip2QFormerPretrain, QFormer, build_qformer_mask

NUM_QUERY = 8
DEC_TOKEN_ID = 2


def build_stage1() -> Blip2QFormerPretrain:
    qformer = QFormer(VOCAB_SIZE, dim=32, depth=4, num_heads=4, image_dim=32, num_query=NUM_QUERY, max_len=32)
    return Blip2QFormerPretrain(tiny_vit(), qformer, DEC_TOKEN_ID, embed_dim=16)


def test_qformer_causal_mask():
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    mask = build_qformer_mask(num_query=2, attention_mask=attention_mask, mode="causal")[0]
    assert mask[:2, :2].all()  # query 互相可见
    assert not mask[:2, 2:].any()  # query 看不到 text
    assert mask[2:, :2].all()  # text 看得到所有 query
    assert mask[3, 2] and not mask[2, 3]  # text 内部因果
    assert not mask[:, 5].any()  # padding 不可见


def test_stage1_forward_backward(images, text):
    model = build_stage1()
    out = model(images, *text)
    for loss in (out.loss_itc, out.loss_itm, out.loss_itg):
        assert torch.isfinite(loss)
    out.loss.backward()
    assert has_grad(model.qformer) and model.qformer.query_tokens.grad is not None
    assert not has_grad(model.visual)  # 图像编码器冻结


def test_stage2_forward_backward_and_generate(images, text):
    model = Blip2ForCausalLM.from_stage1(build_stage1(), tiny_llm())
    out = model(images, *text)
    assert out.logits.shape[1] == NUM_QUERY + text[0].shape[1]
    out.loss.backward()
    assert has_grad(model.llm_proj) and has_grad(model.qformer)
    assert not has_grad(model.llm) and not has_grad(model.visual)

    prompt_ids, prompt_mask = text[0][:, :3], text[1][:, :3]
    assert model.generate(images, prompt_ids, prompt_mask, max_new_tokens=5).shape == (4, 5)
