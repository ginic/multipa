#!/bin/bash

# Data preprocessing
data_dir=data
dataset_cache=dataset_cache

mkdir $data_dir
mkdir $dataset_cache

python preprocess.py --languages en --output_dir $data_dir --num_proc 8 --cache_dir $dataset_cache --clear_cache
