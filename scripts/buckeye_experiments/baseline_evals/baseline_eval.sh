#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=12GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --constraint=avx512
#SBATCH --time 09:00:00
#SBATCH -o %j_baseline_eval.out
#SBATCH --mail-type END

# Evaluation results for our baseline models

EVAL_RESULTS_CSV=data/evaluation_results/aggregate_metrics/baseline_eval.csv
DETAILED_RESULTS_DIR=data/evaluation_results/detailed_predictions
EDIT_DIST_DIR=data/evaluation_results/edit_distances
DATA_DIR=data/buckeye

module load conda/latest
module load uri/main
module load all/eSpeak-NG/1.50-gompi-2020a
conda activate ./env_cuda124

multipa-evaluate --hf_models facebook/wav2vec2-lv-60-espeak-cv-ft ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns  \
 --eval_out $EVAL_RESULTS_CSV \
 --verbose_results_dir $DETAILED_RESULTS_DIR \
 --edit_dist_dir $EDIT_DIST_DIR \
 --no_space --data_dir $DATA_DIR \
 --use_gpu --num_proc 8
