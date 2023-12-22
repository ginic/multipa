#!/bin/bash

# Data preprocessing
mkdir data
mkdir dataset_cache

python preprocess.py --languages en --output_dir data --num_proc 8 --cache_dir dataset_cache --clear_cache
