#!/bin/bash
# Train an audio to IPA transcription speech recognizer

#SBATCH -c 12
#SBATCH --mem=24GB
#SBATCH --p gpu-long
#SBATCH --G 1
#SBATCH --time 12:00:00
#SBACTH -o train.out
#SBATCH --mail-type END

data_dir=data
cache_dir=dataset_cache
vocab_file=data/vocab.en.json

# Wrapper script to train models 
python main.py --language en --train_samples 1000 --test_samples 200 --data_dir $data_dir --num_proc 12 --vocab_file $vocab_file --cache_dir $cache_dir