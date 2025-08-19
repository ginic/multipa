#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=24GB
#SBATCH -p gpu-preempt
#SBATCH -G 4
#SBATCH --constraint=vram40
#SBATCH --nodes=1
#SBATCH --time 3:00:00
#SBATCH -o %j_data_seed_bs64_3.out
#SBATCH --mail-type END

batch_size=4
grad_acc=4
learning_rate=3e-4
model_dir=data/models/data_seed_bs64_3

dataset_cache=dataset_cache
data_dir=data/buckeye


module load conda/latest
conda activate ./env_cuda124

python --version

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 10 --num_proc 8 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    --train_seed 771 \
    buckeye --train_samples 18782 --val_samples 5605 