#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=8GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --time 12:00:00
#SBATCH -o %j_gender_split_70_female_eval.out
#SBATCH --mail-type END


# Evaluation results for our the models that change the data seed only 

EVAL_RESULTS_CSV=data/evaluation_results/aggregate_metrics/gender_split_70_female_eval.csv
DETAILED_RESULTS_DIR=data/evaluation_results/detailed_predictions
DATA_DIR=data/buckeye

module load conda/latest
conda activate ./env

multipa-evaluate --local_models \
 data/models/gender_split_70_female_1/wav2vec2-large-xlsr-buckeye-ipa \
 data/models/gender_split_70_female_2/wav2vec2-large-xlsr-buckeye-ipa \
 data/models/gender_split_70_female_3/wav2vec2-large-xlsr-buckeye-ipa \
 data/models/gender_split_70_female_4/wav2vec2-large-xlsr-buckeye-ipa \
 data/models/gender_split_70_female_5/wav2vec2-large-xlsr-buckeye-ipa \
 --eval_out $EVAL_RESULTS_CSV \
 --verbose_results_dir $DETAILED_RESULTS_DIR \
 --no_space --use_gpu --data_dir $DATA_DIR
