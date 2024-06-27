#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
#PROMPT_VERSION=v1
#MODEL_VERSION="vicuna-v1-3-7b"
#HF_MODEL_NAME=lmsys/vicuna-7b-v1.3
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

################## LLaMA-3 ##################
MODEL_VERSION=Meta-Llama-Guard-2-8B
PROMPT_VERSION=llava_sign_llama_3
HF_MODEL_NAME=meta-llama/${MODEL_VERSION}
################## LLaMA-3 ##################

module load gcc/11.3.0
module load cuda11.8/toolkit/11.8.0-1
CUDA_ID=$CUDA_VISIBLE_DEVICES
source scripts/get_gpu_ids.sh
export CUDA_VISIBLE_DEVICES=$gpu_indices
echo "Setting CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

deepspeed --include localhost:${CUDA_ID} llava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path  ${HF_MODEL_NAME} \
    --version $PROMPT_VERSION \
    --data_path ./phoenix14t/data/anno.finetune.json \
    --s3d_path ./phoenix14t/data/S3D_features.finetune.pkl \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-Meta-Llama-Guard-2-8B-pretrain_phoenix14t_s3d/checkpoint-600/mm_projector.bin \
    --vision_tower openai/clip-vit-base-patch16 \
    --mm_projector_type mlp2x_gelu \
    --mm_hidden_size 832 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 False \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-finetune_lora_phoenix14t_s3d \
    --num_train_epochs 40 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 0 \
    --report_to wandb