#!/bin/bash
#SBATCH --account=msaraclar
#SBATCH --partition=palamut-cuda
#SBATCH --job-name=llava_pose_s2v
#SBATCH --output=slurm/%A.out
#SBATCH --error=slurm/%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH	--time=72:00:00
#SBATCH --chdir=/truba/scratch/msaraclar/Sign_LLaVA
###SBATCH --mal-user= your_email_address
###SBATCH --mail-type=BEGIN,END,FAIL
###SBATCH --mail-type=ALL


### Load modules
module purge
hostname
nvidia-smi

# Load conda environment
eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate llava
which python

# # Change the directory to the sign2vec repository
# cd /truba/scratch/msaraclar/sign2vec
export DATA_DIR="/truba/home/msaraclar/data/How2Sign"
export LOCALDIR="/localscratch/msaraclar/signllava_pose_s2v"
export OUTPUT_DIR="/truba/scratch/msaraclar/models"

echo "local directory: "
echo ${LOCALDIR}
echo "data directory: "
echo ${DATA_DIR}

if [ -d  ${LOCALDIR} ]; then
    echo "Directory exists. Cleaning the directory."
    rm -rf ${LOCALDIR}
else
    echo "Directory does not exist. "
fi

echo "Copying the data to the local scratch. Creating the directory."
# Change the directory to the YASL poses
mkdir -p ${LOCALDIR}
mkdir -p ${LOCALDIR}/output
mkdir -p ${OUTPUT_DIR}/${SLURM_JOB_ID}
cp -r ${DATA_DIR} ${LOCALDIR}

# mv ${LOCALDIR}/How2Sign/sign2vec/sign2vec.train.0.h5 ${LOCALDIR}/How2Sign/sign2vec/sign2vec.train.train.h5
# mv ${LOCALDIR}/How2Sign/sign2vec/sign2vec.dev.0.h5 ${LOCALDIR}/How2Sign/sign2vec/sign2vec.dev.dev.h5
# mv ${LOCALDIR}/How2Sign/sign2vec/sign2vec.test.0.h5 ${LOCALDIR}/How2Sign/sign2vec/sign2vec.test.test.h5

ls ${LOCALDIR}

echo "Finish copying to the local scratch."
date
    
# Set environment variables
export HF_TOKEN="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"
export WANDB_API_KEY="45373c78d8dfa3b3cfd4694785b7f1f0ffbcf570"
export WANDB_ENTITY="jsalt2024-slt"
export WANDB_PROJECT="H2S"
export WANDB_NAME="llama3_8b_pre_10ep_sign2vec"

echo "Start generating pointer file."

CUDA_ID=$CUDA_VISIBLE_DEVICES
source scripts/get_gpu_ids.sh
export CUDA_VISIBLE_DEVICES=$gpu_indices
echo "Setting CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"


########### DO NOT CHANGE ###########
deepspeed --include localhost:${CUDA_ID} --master_port=28313 llava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --gradient_accumulation_steps 1 \
    --yaml_args signllava/configs/truba/how2sign_pretrain_s2v.yaml


cp -r ${LOCALDIR}/output ${OUTPUT_DIR}/${SLURM_JOB_ID}

rm -rf ${LOCALDIR}
echo "Finish training the model."