#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=12GB
#SBATCH -p gpu-preempt
#SBATCH -G 4
#SBATCH --nodes=1
#SBATCH --constraint=vram40
#SBATCH --time 24:00:00
#SBATCH -o %j_hyperparam_tuning_xls-r-300m_4.out
#SBATCH --mail-type END

batch_size=4
grad_acc=2
learning_rate=3e-4
model_dir=data/models/hyperparam_tuning_xls-r-300m_4
dataset_cache=dataset_cache
data_dir=data/buckeye


module load conda/latest
conda activate ./env_cuda124

python --version

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 10 --num_proc 8 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    --base_model facebook/wav2vec2-xls-r-300m \
    --train_seed 42 \
    buckeye --train_samples 4000 --val_samples 5605
