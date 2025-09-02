#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=30GB
#SBATCH -p gpu-preempt
#SBATCH -G 4
#SBATCH --constraint=vram40
#SBATCH --nodes=1
#SBATCH --time 3:00:00
#SBATCH -o %j_train_duration_1600_samples_5.out
#SBATCH --mail-type END

batch_size=4
grad_acc=2
learning_rate=6e-4
model_dir=data/models/train_duration_1600_samples_5
rand_seed=48
train_samples=1600

dataset_cache=dataset_cache
data_dir=data/buckeye


module load conda/latest
conda activate ./env_cuda124

python --version

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 10 --num_proc 8 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    --train_seed $rand_seed \
    buckeye --train_samples $train_samples --val_samples 5605