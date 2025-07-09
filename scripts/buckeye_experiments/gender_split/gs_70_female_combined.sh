#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=12GB
#SBATCH -p gpu-preempt
#SBATCH -G 4
#SBATCH --constraint=vram40
#SBATCH --time 60:00:00
#SBATCH -o %j_combined_gender_split_30_female.out
#SBATCH --mail-type END

batch_size=4
grad_acc=4
learning_rate=3e-4
dataset_cache=dataset_cache
data_dir=data/buckeye

declare -A seeds
seeds[1]=359
seeds[2]=123
seeds[3]=809
seeds[4]=695
seeds[5]=117

module load conda/latest
conda activate ./env_cuda124

python --version
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

for i in "${!seeds[@]}"; do
    seed=${seeds[$i]}
    model_dir=data/models/gender_split_70_female_$i

    echo "Starting training for gender_split_70_female_$i with seed=$seed"

    multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --no_space --cache_dir "$dataset_cache" --use_gpu --num_gpus 4 \
        --num_train_epochs 10 --num_proc 8 --learning_rate $learning_rate --per_device_train_batch_size $batch_size \
        --gradient_accumulation_steps $grad_acc --mask_time_length 4 --train_seed $seed \
        buckeye --train_samples 4000 --val_samples 5605 --percent_female 0.7

    echo "Finished training for gender_split_70_female_$i"
done
