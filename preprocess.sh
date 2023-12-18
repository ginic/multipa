#!bin/bash

# Data preprocessing
mkdir data

python preprocess.py --languages en --output_dir data --num_proc 2
