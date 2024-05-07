#!/bin/bash
# Train an audio to IPA transcription speech recognizer

#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH --p gpu-preempt
#SBATCH --G 1 
#SBATCH --time 12:00:00
#SBATCH -o train_%j.out
#SBATCH --mail-type END

module load miniconda/22.11.1-1
module load cuda/11.3.1

conda activate multipa

data_dir=data
cache_dir=dataset_cache
vocab_file=data/vocab.en.json

# Wrapper script to train models 
multipa-train --language en --train_samples 1000 --test_samples 200 --data_dir $data_dir --num_proc 12 --vocab_file $vocab_file --cache_dir $cache_dir