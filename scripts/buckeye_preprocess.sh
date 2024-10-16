#!/bin/bash
# Script for converting the Buckeye corpus (with train/validation/test splits already defined) to Huggingface format

# TODO Fill in desired paths
raw_buckeye_dir=~/Downloads/BuckeyeData
dataset_cache=dataset_cache
data_dir=data/buckeye

mkdir $data_dir
mkdir $dataset_cache

multipa-preprocess --output_dir "$data_dir" --num_proc 8 --cache_dir "$dataset_cache" --clear_cache buckeye "$raw_buckeye_dir"
