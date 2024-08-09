#!/bin/bash

data_dir=data
cache_dir=dataset_cache
vocab_file=data/vocab.en.json

# Wrapper script to train models 
multipa-train --language en --train_samples 1000 --test_samples 200 --data_dir $data_dir --num_proc 8 --vocab_file $vocab_file --cache_dir $cache_dir