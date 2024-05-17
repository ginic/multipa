#!/bin/bash

#SBATCH -c 12                                                                                                       
#SBATCH --mem=16GB                                                                                                 
#SBATCH -p gpu-preempt
#SBATCH --constraint=[a100|m40|rtx8000]
#SBATCH -G 4
#SBATCH --time 24:00:00
#SBATCH -o train_hyperparam_tuning_3.out
#SBATCH --mail-type END

batch_size=4
grad_acc=4
learning_rate=9e-4

model_dir=data/models/hyperparam_tuning_3

dataset_cache=dataset_cache
data_dir=data/buckeye


module load miniconda/22.11.1-1
module load cuda/11.3.1

conda activate multipa

python --version

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --no_space --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 10 --num_proc 12 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    buckeye --train_samples 4000 --val_samples 5605