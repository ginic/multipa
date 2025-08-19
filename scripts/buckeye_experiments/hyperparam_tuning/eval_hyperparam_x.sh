#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=24GB
#SBATCH -p gpu-preempt
#SBATCH -G 4
#SBATCH --nodes=1
#SBATCH --constraint=vram40
#SBATCH --time 10:00:00
#SBATCH -o %j_hyperparameter_x_models_eval.out
#SBATCH --mail-type END

# Evaluation results for all models

EVAL_RESULTS_CSV=data/evaluation_results/aggregate_metrics/hyperparameter_x_models_eval.csv
DETAILED_RESULTS_DIR=data/evaluation_results/detailed_predictions
DATA_DIR=data/buckeye

module load conda/latest
conda activate ./env_cuda124

# Evaluate all the models
multipa-evaluate --local_models \
 data/models/hyperparam_tuning_xls-r-300m_1/wav2vec2-xls-r-300m-buckeye-ipa \
 data/models/hyperparam_tuning_xls-r-300m_2/wav2vec2-xls-r-300m-buckeye-ipa \
 data/models/hyperparam_tuning_xls-r-300m_3/wav2vec2-xls-r-300m-buckeye-ipa \
 data/models/hyperparam_tuning_xls-r-300m_4/wav2vec2-xls-r-300m-buckeye-ipa \
 data/models/hyperparam_tuning_xls-r-300m_5/wav2vec2-xls-r-300m-buckeye-ipa \
 --eval_out $EVAL_RESULTS_CSV \
 --verbose_results_dir $DETAILED_RESULTS_DIR \
 --no_space --data_dir $DATA_DIR
 --use_gpu