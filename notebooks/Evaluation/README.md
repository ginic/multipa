# Evaluation

This folder contains evaluation notebook for evaluating multiple phonetic transcription models on the **TIMIT** dataset using **IPA (International Phonetic Alphabet)** transcriptions.

## Directory Structure

```text
Evaluation/
├── model_evaluation.ipynb                  # Main notebook to run and compare models on TIMIT
├── timit_to_ipa.py                         # Converts TIMIT .PHN files into IPA using phonecodes
├── Results/
│   ├── timit_dialect_model_comparison.csv              # Metrics per dialect group
│   ├── timit_model_evaluation_summary.csv              # Overall summary of model performance
│   └── timit_subset_with_actual_and_predictions.csv    # Subset predictions + ground truth
```

## Models Evaluated

- `ginic/data_seed_bs64_4_wav2vec2-large-xlsr-53-buckeye-ipa` (Ours - Multipa)
- `ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns` (Taguchi)
- `Allosaurus` (Model: `eng2102`)
- `ZIPA` _(Pending – env & IceFall and K2 issues on Mac)_

## Epitran Dependency For Allosaraus

Epitran or other tools that require `flite` and `lex_lookup`, follow these installation steps:

```bash
git clone http://github.com/festvox/flite
cd flite
./configure && make
sudo make install
cd testsuite
make lex_lookup
sudo cp lex_lookup /usr/local/bin
```
