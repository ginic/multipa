#!/bin/bash

# Data preprocessing
raw_buckeye_dir=~/Downloads/BuckeyeData
dataset_cache=dataset_cache
data_dir=data/buckeye
model_dir=data/buckeye_model

mkdir $data_dir
mkdir $dataset_cache

multipa-preprocess --output_dir "$data_dir" --num_proc 8 --cache_dir "$dataset_cache" --clear_cache buckeye "$raw_buckeye_dir"

multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --no_space --cache_dir "$dataset_cache"  --suffix "_test" buckeye --train_samples 200 --val_samples 100