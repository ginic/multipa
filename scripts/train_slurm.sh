#!/bin/bash
# Train an audio to IPA transcription speech recognizer

#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH -p gpu-preempt
#SBATCH -G 1 
#SBATCH --time 12:00:00
#SBATCH -o train_%j.out
#SBATCH --mail-type END

data_dir=data/buckeye
cache_dir=dataset_cache
vocab_file=data/vocab.en.json

model_dir=data/test_model

module load conda/latest
module load cuda/11.8

conda activate ./env_cuda124

# Wrapper script to train models 
multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --no_space --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 2 --num_proc 12 \
    buckeye --train_samples 1000 --val_samples 200
