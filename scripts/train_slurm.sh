#!/bin/bash
# Train an audio to IPA transcription speech recognizer

#SBATCH -c 8
#SBATCH --nodes=1
#SBATCH --mem=12GB
#SBATCH --constraint=vram16
#SBATCH -p gpu-preempt
#SBATCH -G 1 
#SBATCH --time 12:00:00
#SBATCH -o train_test_%j.out
#SBATCH --mail-type END

data_dir=data/buckeye
cache_dir=dataset_cache
model_dir=data/test_model

module load conda/latest

conda activate ./env_cuda124

# Wrapper script to train models 
multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 2 --num_proc 8 \
    buckeye --train_samples 1000 --val_samples 200
