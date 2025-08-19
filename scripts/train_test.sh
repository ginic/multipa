#!/bin/bash

batch_size=4
grad_acc=4
learning_rate=3e-4
model_dir1=data/models/test1
model_dir2=data/models/test2

dataset_cache=dataset_cache
data_dir=data/buckeye

python --version
echo "How many GPUs found by pytorch?"
python -c "import torch; print(torch.cuda.device_count())"

multipa-train --output_dir "$model_dir1" --data_dir "$data_dir" --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 2 --num_proc 2 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    --train_seed 511 \
    buckeye --train_samples 200 --val_samples 100 \
    --speaker_restriction S01 S04 S08 S09 S12 S21 S06 S11 S13 S15 S28 S30

multipa-train --output_dir "$model_dir2" --data_dir "$data_dir" --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 2 --num_proc 2 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    --base_model facebook/wav2vec2-xls-r-300m \
    --train_seed 511 \
    buckeye --train_samples 200 --val_samples 100 
    