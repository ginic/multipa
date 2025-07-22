#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=12GB
#SBATCH -p gpu-preempt
#SBATCH -G 8
#SBATCH --constraint=vram40
#SBATCH --time 40:00:00
#SBATCH -o %j_combined_data_seeds.out
#SBATCH --mail-type END

batch_size=4
grad_acc=1
learning_rate=3e-4
dataset_cache=dataset_cache
data_dir=data/buckeye

declare -A seeds
seeds[1]=91
seeds[2]=114
seeds[3]=777
seeds[4]=500

module load conda/latest
conda activate ./env_cuda124

python --version

for i in "${!seeds[@]}"; do
    seed=${seeds[$i]}
    model_dir=data/models/data_seed_$i

    echo "Starting training for data_seed_$i with seed=$seed"

    multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --cache_dir "$dataset_cache" --use_gpu --num_train_epochs 10 --num_proc 8 \
        --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
        --train_seed $seed \
        buckeye --train_samples 4000 --val_samples 5605

    echo "Finished training for data_seed_$i"
done
