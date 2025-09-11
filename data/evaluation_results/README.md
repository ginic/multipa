# Buckeye Evaluation
This directory stores evaluation metrics and prediction details for models' performance on the test split of the Buckeye corpus. The code to evaluate these models is found in `scripts/buckeye_experiments` (HuggingFace models) and `notebooks/evaluation_extras.ipynb` (non-HuggingFace models).
For models we've trained on the Buckeye training split, model training code can also be found in `scripts/buckeye_experiments`.

The purposes of the subfolders are as follows:
- aggregate_metrics: "Leaderboard" CSVs that report aggregate (average for error rates, total sums for edit distances) results for each model. Although there are separate CSV files for different batches of experiments, all models are evaluated on the same test set, so they can be combined into a single leaderboard. These files are produced by the `multipa-evaluate` script.
- edit_distances: Have detailed results on the most common phoneme substution, insertion and deletion errors that occur across test set samples for each model. There are 3 CSVs for each model, one for each kind of error. The naming template for files is `{model_name}_{error_type}.csv.`
- detailed_predictions: Contains one file per model with transcription predictions and metrics for every example in the test dataset, one example per line. Files are named like `{model_name}_detailed_predictions.csv`