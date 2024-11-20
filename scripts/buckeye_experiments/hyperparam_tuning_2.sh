#!/bin/bash

#SBATCH -c 12
#SBATCH --mem=24GB
#SBATCH -p gpu-preempt
#SBATCH --constraint=vram48
#SBATCH -G 4 
#SBATCH --time 24:00:00
#SBATCH -o hyperparam_tuning_2.out
#SBATCH --mail-type END

batch_size=4
grad_acc=4
learning_rate=3e-5
model_dir=data/models/hyperparam_tuning_2

dataset_cache=dataset_cache
data_dir=data/buckeye


module load conda/latest
conda activate ./env_cuda121

python --version

pip list

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --no_space --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 10 --num_proc 8 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    --train_seed 13 \
    buckeye --train_samples 4000 --val_samples 5605
