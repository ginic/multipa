#!/bin/bash

#SBATCH -c 12
#SBATCH --mem=8GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --time 12:00:00
#SBATCH -o vary_individuals_eval.out
#SBATCH --mail-type END


# Evaluation results for our the models that change the data seed only 

EVAL_RESULTS_CSV=data/evaluation_results/aggregate_metrics/vary_individuals_eval.csv
DETAILED_RESULTS_DIR=data/evaluation_results/detailed_predictions
DATA_DIR=data/buckeye

module load miniconda/22.11.1-1
conda activate ./env

multipa-evaluate --local_models \
 data/models/vary_individuals_old_only_1/wav2vec2-large-xlsr-buckeye-ipa \
 data/models/vary_individuals_old_only_2/wav2vec2-large-xlsr-buckeye-ipa \
 data/models/vary_individuals_old_only_3/wav2vec2-large-xlsr-buckeye-ipa \
 data/models/vary_individuals_young_only_1/wav2vec2-large-xlsr-buckeye-ipa \
 data/models/vary_individuals_young_only_2/wav2vec2-large-xlsr-buckeye-ipa \
 data/models/vary_individuals_young_only_3/wav2vec2-large-xlsr-buckeye-ipa \
 --eval_out $EVAL_RESULTS_CSV \
 --verbose_results_dir $DETAILED_RESULTS_DIR \
 --no_space --use_gpu --data_dir $DATA_DIR