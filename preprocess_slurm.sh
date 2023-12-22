#!/bin/bash
#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH --p cpu-long
#SBATCH --time 02:00:00
#SBACTH -o preprocessing.out
#SBATCH --mail-type END

# Data preprocessing
mkdir data
mkdir dataset_cache

python preprocess.py --languages en --output_dir data --num_proc 12 --cache_dir dataset_cache --clear_cache
