#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=12GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH --time 08:00:00
#SBATCH -o %j_full_dataset_eval.out
#SBATCH --mail-type END

# Sanity check that models trained correctly. The validation and test sets
# were used in training, so these models should have very good performance metrics, 
# but the results aren't valid as a performance benchmark. 

EVAL_RESULTS_CSV=data/evaluation_results/aggregate_metrics/full_dataset_eval.csv
DETAILED_RESULTS_DIR=data/evaluation_results/detailed_predictions
DATA_DIR=data/buckeye

module load conda/latest
conda activate ./env_cuda124

multipa-evaluate --local_models \
 data/models/full_dataset_train_val_test/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/full_dataset_train_val/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/full_dataset_train_val_bs32/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/full_dataset_train_1/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/full_dataset_train_2/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/full_dataset_train_3/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/full_dataset_train_4/wav2vec2-large-xlsr-53-buckeye-ipa \
 data/models/full_dataset_train_5/wav2vec2-large-xlsr-53-buckeye-ipa \
 --eval_out $EVAL_RESULTS_CSV \
 --verbose_results_dir $DETAILED_RESULTS_DIR \
 --no_space --data_dir $DATA_DIR \
 --use_gpu --num_proc 8