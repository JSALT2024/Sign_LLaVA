#!/usr/bin/env -S bash -e
#SBATCH --job-name=SignLLMpp_pretrain
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --output="/pfs/lustrep2/scratch/project_465000977/gruberiv/logs/output_%x_%j.txt"
#SBATCH --partition=standard-g
#SBATCH --time=48:00:00
#SBATCH --account=project_465000977
export EBU_USER_PREFIX=/project/project_465000977/EasyBuild
module load CrayEnv
module load PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240315
SIF=/pfs/lustrep2/scratch/project_465000977/SIFs/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0-JSALT_17.sif
JOB_DIR=""
DATA_DIR=""
CONFIG_NAME=configs/linear_long.yaml
# MIOpen Error: /MIOpen/src/sqlite_db.cpp:220: Internal error while accessing SQLite database: locking protocol
# Fix ->
# Workaround MIOpen DB issue when using multiple processes
# https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop/blob/main/bonus_material/pytorch_cotainr_container_basics/train_multi_gpu_ddp_torchrun.sh
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
# singularity exec $SIF bash -c "\$WITH_CONDA; pip list"
# singularity exec $SIF bash -c "\$WITH_CONDA; conda list"
cd /scratch/project_465000977/gruberiv/Sign_LLaVA
export PYTHONPATH=$PYTHONPATH:$(pwd)/
# export HF_HUB_CACHE=/pfs/lustrep2/scratch/project_465000977/gruberiv/models/
# export HF_HOME=~/.cache/huggingface
# meta-llama/Meta-Llama-3-8B-Instruct #meta-llama/Meta-Llama-3-70B-Instruct #
RUN_NAME=$(python3 /scripts/config2run.py ${CONFIG_NAME})
export WANDB_API_KEY="c80b9867673b9200b1768293b0b435c170146042"
export WANDB_ENTITY="jsalt2024-slt"
export WANDB_PROJECT="H2S"
export WANDB_NAME=${RUN_NAME}
export LC_ALL=C
# Run the training script
srun singularity exec $SIF bash -c "\$WITH_CONDA; torchrun \
    --standalone \
    --nnodes=$SLURM_JOB_NUM_NODES \
	--nproc_per_node=8 \
    llava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --gradient_accumulation_steps 1 \
    --report_to wandb \
	--run_name ${RUN_NAME} \
    --yaml_args ${CONFIG_NAME} \
    "