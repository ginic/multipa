#!/bin/bash

#SBATCH -c 12
#SBATCH --mem=8GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --time 09:00:00
#SBATCH -o data_seed_eval.out
#SBATCH --mail-type END


# Evaluation results for our the models that change the data seed only 

EVAL_RESULTS_CSV=data/evaluation_results/aggregate_metrics/data_seed_eval.csv
DETAILED_RESULTS_DIR=data/evaluation_results/detailed_predictions
DATA_DIR=data/buckeye

module load miniconda/22.11.1-1
conda activate ./env

multipa-evaluate --local_models \
 data/models/data_seed_1/wav2vec2-large-xlsr-buckeye-ipa \
 data/models/data_seed_2/wav2vec2-large-xlsr-buckeye-ipa \
 data/models/data_seed_3/wav2vec2-large-xlsr-buckeye-ipa \
 data/models/data_seed_4/wav2vec2-large-xlsr-buckeye-ipa \
 --eval_out $EVAL_RESULTS_CSV \
 --verbose_results_dir $DETAILED_RESULTS_DIR \
 --no_space --data_dir $DATA_DIR