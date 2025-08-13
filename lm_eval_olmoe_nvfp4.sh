#!/bin/bash
# Evaluate NVFP4 GPTQ-quantized OLMoE model on GSM8K with A4W4 settings
MODEL_NAME=allenai/OLMoE-1B-7B-0924-Instruct
QUANT_MODEL_PATH=autogptq_allenai/OLMoE-1B-7B-0924-Instruct-nvfp4

# Use NVFP4 activation quantization
echo "Evaluating $MODEL_NAME from $QUANT_MODEL_PATH on GSM8K"
QUANT_MODE=Dynamic_Block CUDA_VISIBLE_DEVICES=0 python lm_eval_gptq.py \
    --model_name $MODEL_NAME \
    --quant_model_path $QUANT_MODEL_PATH \
    --is_quantized \
    --tasks gsm8k

