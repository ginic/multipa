#!/bin/bash

# Sanity check that models trained correctly.

EVAL_RESULTS_CSV=data/edit_dist_eval_test/aggregate_metrics/full_dataset_eval.csv
DETAILED_RESULTS_DIR=data/edit_dist_eval_test/detailed_predictions
EDIT_DIST_DIR=data/edit_dist_eval_test/edit_distances
DATA_DIR=data/buckeye

multipa-evaluate --hf_models \
 ginic/full_dataset_train_1_wav2vec2-large-xlsr-53-buckeye-ipa \
 ginic/full_dataset_train_2_wav2vec2-large-xlsr-53-buckeye-ipa \
 ginic/full_dataset_train_3_wav2vec2-large-xlsr-53-buckeye-ipa \
 ginic/full_dataset_train_4_wav2vec2-large-xlsr-53-buckeye-ipa \
 ginic/full_dataset_train_5_wav2vec2-large-xlsr-53-buckeye-ipa \
 --eval_out $EVAL_RESULTS_CSV \
 --verbose_results_dir $DETAILED_RESULTS_DIR \
 --edit_dist_dir $EDIT_DIST_DIR \
 --no_space --data_dir $DATA_DIR \
 --use_gpu --num_proc 8