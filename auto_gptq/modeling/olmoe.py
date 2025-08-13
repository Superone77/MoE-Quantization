from ._base import BaseGPTQForCausalLM_mixed_precision


class OlmoeGPTQForCausalLM(BaseGPTQForCausalLM_mixed_precision):
    """GPTQ wrapper for OLMoE models."""

    model_type = "olmoe"
    layer_type = "OlmoeDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]

    moe_w13_list = []
    for i in range(8):
        moe_w13_list.append(f"block_sparse_moe.experts.{i}.w1")
        moe_w13_list.append(f"block_sparse_moe.experts.{i}.w3")

    moe_w2_list = []
    for i in range(8):
        moe_w2_list.append(f"block_sparse_moe.experts.{i}.w2")

    inside_layer_modules = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        moe_w13_list,
        moe_w2_list,
    ]


__all__ = ["OlmoeGPTQForCausalLM"]
