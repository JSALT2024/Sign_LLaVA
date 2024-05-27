# Get the number of GPUs using nvidia-smi
num_gpus=$(nvidia-smi -L | wc -l)

# Initialize an empty string to store the GPU indices
gpu_indices=""

# Loop through each GPU index and append to the string
for ((i=0; i<num_gpus; i++)); do
    if [[ $i -lt $((num_gpus - 1)) ]]; then
        gpu_indices+="${i},"
    else
        gpu_indices+="${i}"
    fi
done

# Output the result which can be used to set CUDA_VISIBLE_DEVICES
echo $gpu_indices