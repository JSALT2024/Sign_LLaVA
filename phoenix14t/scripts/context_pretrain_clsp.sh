#!/bin/bash
# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
MODEL_VERSION=Meta-Llama-Guard-2-8B #Llama-2-7b-chat-hf
#MODEL_VERSION=Meta-Llama-3-70B-Instruct
PROMPT_VERSION=llava_sign_llama_3 #llava_llama_2
########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
#module load gcc/11.3.0
module load cuda/12.1
cd /export/fs04/xzhan138/Sign_LLaVA

CUDA_ID=$CUDA_VISIBLE_DEVICES
source scripts/get_gpu_ids.sh
export CUDA_VISIBLE_DEVICES=$gpu_indices
echo "Setting CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
########### DO NOT CHANGE ###########
deepspeed --include localhost:${CUDA_ID} --master_port=29501 llava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/${MODEL_VERSION} \
    --version $PROMPT_VERSION \
    --data_path ./phoenix14t/data/anno.pretrain.json \
    --s3d_path ./phoenix14t/data/S3D_features.pretrain.pkl \
    --vision_tower openai/clip-vit-base-patch16 \
    --tune_mm_mlp_adapter True \
    --mm_projector_type mlp2x_gelu \
    --mm_hidden_size 832 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-pretrain_phoenix14t_s3d \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 2e-3 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to wandb