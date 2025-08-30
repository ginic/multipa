#!/bin/bash

#SBATCH -c 4
#SBATCH --mem=40GB
#SBATCH -p gpu-preempt
#SBATCH -G 4
#SBATCH --constraint=vram40
#SBATCH --nodes=1
#SBATCH --time 8:00:00
#SBATCH -o %j_full_dataset_train_val.out
#SBATCH --mail-type END

batch_size=4
grad_acc=4
learning_rate=3e-4
model_dir=data/models/full_dataset_train_val

dataset_cache=dataset_cache
data_dir=data/buckeye


module load conda/latest
conda activate ./env_cuda124

python --version

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 10 --num_proc 4 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    buckeye --train_samples 30000 --val_samples 5605 \
    --percent_female -1 \
    --use_val_split_in_training