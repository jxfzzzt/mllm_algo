# mllm_algo

多模态大模型经典算法的 PyTorch 实现：**CLIP、ALBEF、BLIP、BLIP-2、Flamingo、LLaVA、Llama 3.2-Vision**。

每类算法只依赖 PyTorch，每个算法一个包，结构保留论文的核心设计，次要的工程细节都做了简化，并在注释里说明。

## 目录结构

```
mllm_algo/
├── README.md
├── pyproject.toml
├── mllm_algo/
│   ├── common/                  # 所有算法共用的基础组件
│   │   ├── attention.py         #   多头注意力 / FFN / Transformer Block / mask 工具函数
│   │   ├── vision.py            #   ViT 图像编码器
│   │   ├── language_model.py    #   最小 decoder-only LLM (接口与 HuggingFace 一致)
│   │   └── losses.py            #   对比损失 / 语言模型损失 / ITM 难负样本采样
│   ├── clip/
│   │   ├── text_encoder.py      #   因果 Transformer 文本编码器, 取 [EOS]
│   │   └── model.py             #   CLIP: 对比学习 + 零样本分类
│   ├── albef/
│   │   ├── text_encoder.py      #   BERT 拆成两半: 前 6 层文本编码器, 后 6 层融合编码器
│   │   ├── masking.py           #   BERT 式 15% 随机 mask
│   │   └── model.py             #   ALBEF 预训练: ITC + ITM + MLM, 动量蒸馏 + 特征队列
│   ├── blip/
│   │   ├── med.py               #   MED: 一个文本模型切换 text / multimodal / decoder 三种模式
│   │   └── model.py             #   BLIP 预训练: ITC(动量蒸馏+队列) + ITM + LM
│   ├── blip2/
│   │   ├── qformer.py           #   Q-Former + 三种 attention mask
│   │   └── model.py             #   Stage 1: ITC + ITM + ITG;  Stage 2: 接冻结 LLM 生成
│   ├── flamingo/
│   │   ├── perceiver.py         #   Perceiver Resampler
│   │   ├── gated_xattn.py       #   门控交叉注意力 + "只看最近一张图" 的 media mask
│   │   └── model.py             #   Flamingo: 在冻结 LM 层之间插入 GATED XATTN
│   ├── llava/
│   │   ├── projector.py         #   linear (v1) / mlp2x_gelu (v1.5)
│   │   └── model.py             #   LLaVA: <image> 占位符替换成视觉 token + 两阶段训练
│   └── llama32_vision/
│       ├── image_processing.py  #   高分辨率图像按宽高比切成最多 4 个 tile
│       ├── vision_encoder.py    #   ViT-H + tile 位置编码 + 门控 global 层 + 多层特征拼接
│       ├── text_model.py        #   Llama 3.1 (RMSNorm/RoPE/GQA/SwiGLU) + 门控 cross-attention 层
│       └── model.py             #   完整模型 + 文本到图像的 cross-attention mask
├── tests/                       # 每个算法的冒烟测试 (小尺寸模型, CPU 秒级跑完)
└── refs/                        # 各算法对应的原始论文 PDF, 见 refs/README.md
```

## 算法对比

| | 视觉编码器 | 连接方式 | 语言侧 | 训练目标 | 冻结 |
|---|---|---|---|---|---|
| **CLIP** | ViT | 无 (双塔, 各自投影到同一空间) | 文本编码器 | 对比学习 | 无, 从头训练 |
| **ALBEF** | ViT | BERT 后 6 层加 cross-attention 做融合 | BERT (前 6 层纯文本) | ITC + ITM + MLM, 动量蒸馏 | 无 |
| **BLIP** | ViT | 文本模型中的 cross-attention | MED (编码器/解码器共享参数) | ITC + ITM + LM | 无 |
| **BLIP-2** | ViT-g | **Q-Former** (32 个 query) + Linear, 作为 LLM 的前缀 | OPT / FlanT5 | Stage 1: ITC+ITM+ITG; Stage 2: LM | ViT、LLM |
| **Flamingo** | NFNet | **Perceiver Resampler** + **门控 cross-attention** 插入 LM | Chinchilla | LM (图文交错数据) | 视觉编码器、LM |
| **LLaVA** | CLIP ViT-L | **Linear / MLP**, 视觉 token 直接拼进输入序列 | Vicuna / LLaMA | LM (指令数据) | ViT; Stage 1 冻结 LLM |
| **Llama 3.2-Vision** | ViT-H/14 + 8 层门控 global 层, 高分辨率切 tile, 多层特征拼接 | Linear + 每 4 层插 1 层**门控 cross-attention** | Llama 3.1 | LM | Llama 3.1 (视觉编码器和 adapter 都训练) |

视觉信息进入 LLM 有两条路线：
- **拼进输入序列**：BLIP-2、LLaVA。结构简单，但序列长度会随图像变长。BLIP-2 用 Q-Former 压缩到 32 个 token，LLaVA 直接用全部 patch。
- **通过 cross-attention 注入**：Flamingo、Llama 3.2-Vision。LLM 的序列长度不变，适合多图和高分辨率输入，但要改 LLM 的结构。Llama 3.2-Vision 冻结了 Llama 3.1，纯文本输入时跳过 cross-attention 层，输出和 Llama 3.1 完全一致。

## 运行测试

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

## 使用示例

```python
import torch
from mllm_algo.common import CausalLM, VisionTransformer
from mllm_algo.llava import IMAGE_TOKEN_INDEX, LlavaForCausalLM

vit = VisionTransformer(image_size=224, patch_size=14, dim=1024, depth=24, num_heads=16)
llm = CausalLM(vocab_size=32000, dim=4096, depth=32, num_heads=32)  # 也可以直接传 HuggingFace 的 LLaMA
model = LlavaForCausalLM(vit, llm, projector_type="mlp2x_gelu")

model.set_train_stage(1)  # 特征对齐: 只训练 projector
model.set_train_stage(2)  # 指令微调: 训练 projector + LLM
```

`common.CausalLM` 的接口（`get_input_embeddings`、`inputs_embeds`、`labels`、`generate`）和 HuggingFace 保持一致，BLIP-2 Stage 2 和 LLaVA 可以直接换成真实的 LLM。Flamingo 要改 LM 内部的层，所以依赖 `CausalLM` 的 `layers` 结构。

## 简化部分

- 所有 Transformer 统一用 Pre-Norm；Q-Former 保留 BERT 的 Post-Norm。
- 没有实现分布式训练里的 `all_gather`（CLIP / BLIP / BLIP-2 用它跨卡收集负样本）。
- BLIP 的 CapFilt 是数据清洗流程，不涉及模型结构，没有实现。
- Flamingo 中 `<image>`、`<|endofchunk|>` 的词向量在原论文里可训练，这里复用了冻结的词表。
- LLaVA 只支持每条样本最多一张图；没有实现 LLaVA-NeXT 的 AnyRes 高分辨率切图。
- Llama 3.2-Vision：没有实现 Llama 3.1 的 RoPE 频率缩放，也没有把每个 tile 的 patch 数补齐到 8 的倍数；图像预处理省略了归一化。
- 生成只实现了贪心解码，没有 KV cache。
