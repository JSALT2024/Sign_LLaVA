#!/bin/bash
module load cuda/12.1
cd /home/kduh/src/signlang/Sign_LLaVA

CUDA_ID=$CUDA_VISIBLE_DEVICES
source scripts/get_gpu_ids.sh
export CUDA_VISIBLE_DEVICES=$gpu_indices
echo "Setting CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
########### DO NOT CHANGE ###########
deepspeed --include localhost:${CUDA_ID} --master_port=29515 llava/eval/run_signllava.py \
    --deepspeed ./scripts/zero2.json \
    --yaml_args signllava/configs/eval.yaml
