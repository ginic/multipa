#!/bin/bash

#SBATCH -c 12
#SBATCH --mem=16GB
#SBATCH -p gpu-preempt
#SBATCH -G 4
#SBATCH --constraint=vram40
#SBATCH --time 20:00:00
#SBATCH -o train_gender_split_70_female_2.out
#SBATCH --mail-type END

batch_size=4
grad_acc=2
learning_rate=3e-4
model_dir=data/models/gender_split_70_female_2

dataset_cache=dataset_cache
data_dir=data/buckeye


module load miniconda/22.11.1-1
conda activate ./env

python --version

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --no_space --cache_dir "$dataset_cache" --use_gpu --num_gpus 4 --num_train_epochs 10 --num_proc 12 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    --train_seed 131 \
    buckeye --train_samples 4000 --val_samples 5605 --percent_female 0.7
