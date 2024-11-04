#!/bin/bash

#SBATCH -c 12
#SBATCH --mem=8GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --time 09:00:00
#SBATCH -o baseline_eval.out
#SBATCH --mail-type END


# Evaluation results for our baseline models, the Taguchi 2K multilingual model 
# and our best fine-tuned hyper param model

EVAL_RESULTS_CSV=data/evaluation_results/aggregate_metrics/baseline_eval.csv
DETAILED_RESULTS_DIR=data/evaluation_results/detailed_predictions
DATA_DIR=data/buckeye

module load miniconda/22.11.1-1
conda activate ./env

multipa-evaluate --hf_models ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa-plus-2000 \
 --local_models data/models/hyperparam_tuning_1/wav2vec2-large-xlsr-buckeye-ipa \
 --eval_out $EVAL_RESULTS_CSV \
 --verbose_results_dir $DETAILED_RESULTS_DIR \
 --no_space --data_dir $DATA_DIR