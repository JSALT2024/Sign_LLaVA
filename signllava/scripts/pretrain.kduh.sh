#!/bin/bash
module load cuda/12.1
cd /home/kduh/src/signlang/Sign_LLaVA

CUDA_ID=$CUDA_VISIBLE_DEVICES
source scripts/get_gpu_ids.sh
export CUDA_VISIBLE_DEVICES=$gpu_indices
echo "Setting CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
########### DO NOT CHANGE ###########
deepspeed --include localhost:${CUDA_ID} --master_port=29501 llava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --bf16 True \
    --gradient_accumulation_steps 16 \
    --output_dir signllava/checkpoints/llama3_70b_4reps_context2_prelude2 \
    --report_to wandb \
    --yaml_args signllava/configs/s2v_mae_dino_context3.pretrain.multih5.yaml
