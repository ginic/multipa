#!/bin/bash
# Preprocess data for training speech recognition model

#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH -p cpu-preempt
#SBATCH --time 02:00:00
#SBATCH -o preprocessing_%j.out
#SBATCH --mail-type END

# Load slurm stuff
module load miniconda/22.11.1-1
conda activate multipa

# Data preprocessing
data_dir=data
dataset_cache=dataset_cache

mkdir $data_dir
mkdir $dataset_cache

python preprocess.py --languages en --output_dir $data_dir --num_proc 12 --cache_dir $dataset_cache --clear_cache
