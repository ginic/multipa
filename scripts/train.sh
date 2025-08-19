#!/bin/bash

data_dir=data/buckeye
cache_dir=dataset_cache
model_dir=data/test_model

# Wrapper script to train models
multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --cache_dir "$dataset_cache" --num_train_epochs 2 --num_proc 8 \
    buckeye --train_samples 1000 --val_samples 200