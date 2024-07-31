#!/bin/bash
CUDA_ID=$CUDA_VISIBLE_DEVICES
source scripts/get_gpu_ids.sh
export CUDA_VISIBLE_DEVICES=$gpu_indices
echo "Setting CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export WANDB_WATCH=all

########### DO NOT CHANGE ###########
deepspeed --include localhost:${CUDA_ID} --master_port=29515 llava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --gradient_accumulation_steps 1 \
    --yaml_args signllava/configs/truba/how2sign_finetune_s2v.yaml
