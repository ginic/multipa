#!/bin/bash

#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH --p gpu-preempt
#SBATCH --G 2 
#SBATCH --time 12:00:00
#SBATCH -o train_hyperparam_tuning_1.out
#SBATCH --mail-type END

batch_size=4
grad_acc=8
learning_rate="3e-4"
model_dir=data/models/hyperparam_tuning_1

dataset_cache=dataset_cache
data_dir=data/buckeye


module load miniconda/22.11.1-1
module load cuda/11.3.1

conda activate multipa

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --no_space --cache_dir "$dataset_cache" --use_gpu --num_training_epochs 2 --num_proc 12 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $8 \
    buckeye --train_samples 2000 --val_samples 5605