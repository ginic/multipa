#!/bin/bash

# Data preprocessing
data_dir=data/buckeye
model_dir=data/buckeye_model

multipa-evaluate --hf_models ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa-plus-2000 \
 --local_models data/buckeye_model/wav2vec2-large-xlsr-buckeye-ipa_test \
 --eval_out data/test_eval/buckeye_eval.csv \
 --verbose_results_dir data/test_eval/detailed_results --no_space --data_dir $data_dir