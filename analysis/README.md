# Duration PER Sample Count Analysis

`duration_per_analysis.py` plots the relationship between **audio duration** and **Phone Error Rate (PER)** using evaluation prediction files.

The script loads prediction CSV files from multiple training runs, selects the run with the **lowest mean PER**, and generates summary statistics and plots showing how PER changes with duration.

---

## Inputs

The script reads prediction files from: `data/evaluation_results/detailed_predictions/`

Expected files:

- ginic*full_dataset_train_1*...
- ginic*full_dataset_train_2*...
- ginic*full_dataset_train_3*...
- ginic*full_dataset_train_4*...
- ginic*full_dataset_train_5*...

Each file must contain the following columns:

- duration
- phone_error_rates

---

## Outputs

All outputs are saved in: `analysis/output/`

### CSV files

| File           | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| 0_12.csv       | Summary statistics using **1 second bins** from 0–12 seconds   |
| 0_12_finer.csv | Summary statistics using **0.1 second bins** from 0–12 seconds |

Each CSV contains:

- bin_start
- bin_end
- mean_duration
- mean_phone_error_rate
- sample_count

### Plots

| File                          | Description                                                |
| ----------------------------- | ---------------------------------------------------------- |
| 0_1.png                       | PER and sample count vs duration (0–1 seconds, 0.1s bins)  |
| 0_1_finer.png                 | PER and sample count vs duration (0–1 seconds, 0.01s bins) |
| 0_12.png                      | PER and sample count vs duration (0–12 seconds, 1s bins)   |
| 0_12_finer.png                | PER and sample count vs duration (0–12 seconds, 0.1s bins) |
| moving_average_increasing.png | Mean PER for samples with duration ≥ threshold             |
| moving_average_decreasing.png | Mean PER for samples with duration ≤ threshold             |
