#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=16GB
#SBATCH -p gpu-preempt
#SBATCH -G 4
#SBATCH --constraint=vram40
#SBATCH --time 20:00:00
#SBATCH -o %j_vary_individuals_young_only_1.out
#SBATCH --mail-type END

batch_size=4
grad_acc=4
learning_rate=3e-4
model_dir=data/models/vary_individuals_young_only_1

dataset_cache=dataset_cache
data_dir=data/buckeye


module load conda/latest
conda activate ./env_cuda124

python --version


multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --no_space --cache_dir "$dataset_cache" --use_gpu --num_gpus 4 --num_train_epochs 10 --num_proc 8 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    --train_seed 359 \
    buckeye --train_samples 4000 --val_samples 5605 \
    --speaker_restriction S01 S04 S08 S09 S12 S21 S06 S11 S13 S15 S28 S30
