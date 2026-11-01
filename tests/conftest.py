"""测试用的小尺寸模型和假数据, 保证每个测试在 CPU 上几秒内跑完。"""
import pytest
import torch

from mllm_algo.common import CausalLM, VisionTransformer

VOCAB_SIZE = 100
BATCH_SIZE = 4
TEXT_LEN = 12


def tiny_vit() -> VisionTransformer:
    return VisionTransformer(image_size=32, patch_size=8, dim=32, depth=3, num_heads=4)  # 16 个 patch


def tiny_llm() -> CausalLM:
    return CausalLM(VOCAB_SIZE, dim=48, depth=4, num_heads=4, max_len=128)


def has_grad(module: torch.nn.Module) -> bool:
    return any(p.grad is not None for p in module.parameters())


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture
def images() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, 3, 32, 32)


@pytest.fixture
def text():
    """右 padding 的文本, 第 0 条样本较短。"""
    input_ids = torch.randint(5, VOCAB_SIZE, (BATCH_SIZE, TEXT_LEN))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, 8:] = 0
    input_ids[0, 8:] = 0
    return input_ids, attention_mask
