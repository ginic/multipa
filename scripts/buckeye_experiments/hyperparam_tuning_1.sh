#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=12GB
#SBATCH -p gpu-preempt
#SBATCH -G 2
#SBATCH --constraint=[a100|m40|rtx8000]
#SBATCH --time 24:00:00
#SBATCH -o train_hyperparam_tuning_1.out
#SBATCH --mail-type END

batch_size=2
grad_acc=16
learning_rate=3e-4
model_dir=data/models/hyperparam_tuning_1

dataset_cache=dataset_cache
data_dir=data/buckeye


module load miniconda/22.11.1-1
module load cuda/11.8.0

conda activate multipa

python --version

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --no_space --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 10 --num_proc 8 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    buckeye --train_samples 4000 --val_samples 5605
