#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=12GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH --time 08:00:00
#SBATCH -o %j_train_duration_400_hyperparam_eval.out
#SBATCH --mail-type END

# Evaluation results for all models

EVAL_RESULTS_CSV=data/evaluation_results/aggregate_metrics/train_duration_400_hyperparam_eval.csv
DETAILED_RESULTS_DIR=data/evaluation_results/detailed_predictions
DATA_DIR=data/buckeye

module load conda/latest
conda activate ./env_cuda124

# Evaluate all the models
multipa-evaluate --local_models \
 data/models/train_duration_400_samples_hyperparams_1/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/train_duration_400_samples_hyperparams_2/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/train_duration_400_samples_hyperparams_3/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/train_duration_400_samples_hyperparams_4/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/train_duration_400_samples_hyperparams_5/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/train_duration_400_samples_hyperparams_6/wav2vec2-large-xlsr-53-buckeye-ipa \
 --eval_out $EVAL_RESULTS_CSV \
 --verbose_results_dir $DETAILED_RESULTS_DIR \
 --no_space --data_dir $DATA_DIR \
 --use_gpu --num_proc 8