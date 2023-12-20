#!/bin/bash

# Wrapper script to train models 
python main.py --language en --train_samples 1000 --test_samples 200 --data_dir data --num_proc 8