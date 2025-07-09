#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=8GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --time 09:00:00
#SBATCH -o %j_wav2vec2-xlsr_eval.out
#SBATCH --mail-type END


# Evaluation results for our baseline models, the Taguchi 2K multilingual model
# and our best fine-tuned hyper param model

EVAL_RESULTS_CSV=data/evaluation_results/aggregate_metrics/wav2vec2-xlsr-53-espeak-cv-ft_baseline_eval.csv
DETAILED_RESULTS_DIR=data/evaluation_results/detailed_predictions
DATA_DIR=data/buckeye

module load conda/latest
module load uri/main
module load all/eSpeak-NG/1.50-gompi-2020a
conda activate ./env_cuda124

multipa-evaluate --hf_models facebook/wav2vec2-xlsr-53-espeak-cv-ft \
 --eval_out $EVAL_RESULTS_CSV \
 --verbose_results_dir $DETAILED_RESULTS_DIR \
 --no_space --data_dir $DATA_DIR
