#!/bin/bash
CUDA_ID=$CUDA_VISIBLE_DEVICES
source scripts/get_gpu_ids.sh
export CUDA_VISIBLE_DEVICES=$gpu_indices
echo "Setting CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
########### DO NOT CHANGE ###########
#deepspeed --include localhost:${CUDA_ID} --master_port=29515 llava/eval/run_signllava.py \
# --deepspeed ./scripts/zero2.json \
python llava/eval/run_signllava_multi.py --yaml_args signllava/configs/truba/how2sign_finetune_s2v.yaml