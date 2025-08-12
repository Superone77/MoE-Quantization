import random
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM_mixed_precision, BaseQuantizeConfig_mixed_precision


def get_wikitext2(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
    if split == "train":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    else:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "".join([" \n" if s == "" else s for s in data["text"][:1000]])
    enc = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset


def build_bits_dict(config, attn_bits: int, expert_bits: int):
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layers", 0))
    num_experts = getattr(config, "num_local_experts", 8)
    bits = {}
    for layer in range(num_layers):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            bits[f"model.layers.{layer}.self_attn.{proj}"] = attn_bits
        for expert in range(num_experts):
            for part in ["w1", "w2", "w3"]:
                bits[f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{part}"] = expert_bits
    return bits


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--attn_bits", type=int, default=4)
    parser.add_argument("--expert_bits", type=int, default=4)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    bits = build_bits_dict(config, args.attn_bits, args.expert_bits)
    quantize_config = BaseQuantizeConfig_mixed_precision(
        bits=bits,
        group_size=args.group_size,
        desc_act=False,
        model_file_base_name=f"{args.model_name.split('/')[-1]}-nvfp4",
        quant_type="nvfp4",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoGPTQForCausalLM_mixed_precision.from_pretrained(
        args.model_name,
        quantize_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    dataset = get_wikitext2(tokenizer, seqlen=4096, nsamples=args.nsamples)
    model.quantize(dataset)
    quant_path = f'autogptq_{args.model_name}-nvfp4'
    model.save_quantized(quant_path)
    print(f"Quantized model saved to {quant_path}")


if __name__ == "__main__":
    main()
