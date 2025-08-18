#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=12GB
#SBATCH -p gpu-preempt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint=vram40
#SBATCH --time 24:00:00
#SBATCH -o %j_combined_vary_individuals_old_only.out
#SBATCH --mail-type END

batch_size=4
grad_acc=4
learning_rate=3e-4
dataset_cache=dataset_cache
data_dir=data/buckeye

declare -A seeds
seeds[1]=979
seeds[2]=113
seeds[3]=942

speakers=(S02 S05 S07 S14 S16 S17 S03 S10 S19 S22 S24)

module load conda/latest
conda activate ./env_cuda124

python --version
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

for i in "${!seeds[@]}"; do
    seed=${seeds[$i]}
    model_dir=data/models/vary_individuals_old_only_$i

    echo "Starting training for vary_individuals_old_only_$i with seed=$seed"

    multipa-train --output_dir "$model_dir" --data_dir "$data_dir" --cache_dir "$dataset_cache" --use_gpu --num_gpus 4 \
        --num_train_epochs 10 --num_proc 8 --learning_rate $learning_rate \
        --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
        --train_seed $seed \
        buckeye --train_samples 4000 --val_samples 5605 --speaker_restriction "${speakers[@]}"

    echo "Finished training for vary_individuals_old_only_$i"
done
