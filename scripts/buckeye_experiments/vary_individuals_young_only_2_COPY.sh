#!/bin/bash

#SBATCH -c 12
#SBATCH --mem=24GB
#SBATCH -p gpu-preempt,gpu
#SBATCH -G 1
#SBATCH --constraint=vram40
#SBATCH --time 24:00:00
#SBATCH -o train_vary_individuals_young_only_2_%j.out
#SBATCH --mail-type END
#SBATCH --export=None

batch_size=4
grad_acc=4
learning_rate=3e-4
model_dir=data/models/vary_individuals_young_only_2

dataset_cache=dataset_cache
data_dir=data/buckeye

export HF_DATASETS_CACHE=$dataset_cache

source /work/pi_vcpartridge_umass_edu/georgia_test/venv/bin/activate
python --version
echo "How many GPUs found by pytorch?"
python -c "import torch; print(torch.cuda.device_count())"

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --no_space --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 10 --num_proc 12 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    --train_seed 511 \
    buckeye --train_samples 4000 --val_samples 5605 \
    --speaker_restriction S01 S04 S08 S09 S12 S21 S06 S11 S13 S15 S28 S30
