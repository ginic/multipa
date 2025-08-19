"""Evaluates given models on test data, writing evaluation metrics in a summary CSV file.
Detailed results on the test data may also be written if desired.

Currently only Buckeye data is supported for evaluation.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import datasets
import evaluate

import pandas as pd
import panphon.distance
import transformers
import torch

from multipa.data_utils import load_buckeye_split, clean_text, EMPTY_TRANSCRIPTION

PHONE_ERRORS_EVALUATOR = evaluate.load("ginic/phone_errors")
DETAILED_PREDICTIONS_CSV_SUFFIX = "detailed_predictions.csv"
HALLUCINATIONS_SUFFIX = "hallucinations.csv"


class ModelEvaluator:
    model_key = "model"
    # Metric names that will become column headers
    per_key = "mean_phone_error_rate"
    pfer_key = "mean_phone_feature_error_rate"
    fer_key = "mean_feature_error_rate"
    phone_hallucinations_key = "total_phone_hallucinations"

    def __init__(self):
        self.distance_computer = panphon.distance.Distance()
        # Final results will have these keys
        # {model name -> {metric_key: metric_value}}
        self.results_to_write = defaultdict(dict)

    def eval_non_empty_transcriptions(self, model_name, predictions, references):
        """Compare the predictions and gold-standard references for a model,
        then add results to model results tracker.
        Returns the full detailed evaluation results as a dictionary.
        """
        metrics = PHONE_ERRORS_EVALUATOR.compute(predictions=predictions, references=references)
        for k in [ModelEvaluator.per_key, ModelEvaluator.pfer_key, ModelEvaluator.fer_key]:
            self.results_to_write[model_name][k] = metrics[k]
        return metrics

    def eval_empty_transcriptions(self, model_name, predictions):
        """Count number of phone hallucinations for this model and save to write later.
        Returns the detailed evaluation results as a list with one entry per prediction.
        """
        phone_lengths = [len(self.distance_computer.fm.ipa_segs(p)) for p in predictions]
        total_phone_hallucinations = sum(phone_lengths)
        self.results_to_write[model_name][ModelEvaluator.phone_hallucinations_key] = total_phone_hallucinations
        return phone_lengths

    def to_csv(self, csv_path):
        """Write the evaluation results stored in this object to the specified CSV file"""
        df = pd.DataFrame.from_dict(self.results_to_write, orient="index")
        df.index.name = ModelEvaluator.model_key
        df.to_csv(csv_path)


def preprocess_test_data(test_dataset: datasets.Dataset, is_remove_space: bool = False):
    """
    Filters the test dataset into examples with non-empty and empty transcriptions,
    since they should be evaluated separately.
    Also performs additional text cleaning if specified.

    Args:
        test_dataset: Huggingface dataset you'll use for evaluation
        is_remove_space: Filter out spaces in IPA strings if true

    Returns:
        non_empty_transcriptions_dataset, empty_transcriptions_dataset: a tuple of Huggingface datasets
    """
    # Set sampling rate to 16K
    input_data = test_dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000)).map(
        lambda x: clean_text(x, is_remove_space=is_remove_space)
    )

    empty_test_data = input_data.filter(lambda x: x["ipa"] == EMPTY_TRANSCRIPTION)
    non_empty_test_data = input_data.filter(lambda x: x["ipa"] != EMPTY_TRANSCRIPTION)

    return non_empty_test_data, empty_test_data


def clean_predictions_batch(predictions_batch, target_key: str = "text", is_remove_space: bool = False) -> list[str]:
    """Convenience function for removing spaces from model output
    that matches the way spaces are removed from training data.
    This helps ensure that spaces are handled the same way in both predictions
    and evaluation data.

    Args:
        predictions_batch: Iterable of dictionaries, the output of the model pipeline
        target_key: str, key for the transcription output in a single prediction. Defaults to "text".
        is_remove_space: bool, whether to remove spaces from model predictions. Defaults to False.

    Returns:
        A list of predicted transcriptions with spaces removed as desired
    """
    clean_batch = map(lambda x: clean_text(x, text_key=target_key, is_remove_space=is_remove_space), predictions_batch)
    return [d[target_key] for d in clean_batch]


def get_torch_device(use_gpu: bool = False):
    """Return the torch.device to load the model to.

    Args:
        use_gpu (bool, optional): True if the user desires GPU for model inference. Defaults to False.
    """
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main(
    input_data: datasets.Dataset,
    eval_csv: Union[Path, str],
    local_models: Optional[list[Path]] = None,
    hf_models: Optional[list[str]] = None,
    verbose_results_dir: Optional[Path] = None,
    is_remove_space: bool = False,
    use_gpu: bool = False,
):
    if local_models is None:
        local_models = []
    if hf_models is None:
        hf_models = []

    non_empty_test_data, empty_test_data = preprocess_test_data(input_data, is_remove_space)
    print("Number of test examples with NON-empty transcriptions:", len(non_empty_test_data))
    print(non_empty_test_data)

    print("Number of test examples with empty transcriptions:", len(empty_test_data))
    print(empty_test_data)

    model_eval_tracker = ModelEvaluator()

    selected_torch_device = get_torch_device(use_gpu)

    for model in local_models + hf_models:
        print("Evaluating model:", model)
        pipe = transformers.pipeline("automatic-speech-recognition", model=model, device=selected_torch_device)
        predictions = clean_predictions_batch(pipe(non_empty_test_data["audio"]), is_remove_space=is_remove_space)

        metrics = model_eval_tracker.eval_non_empty_transcriptions(model, predictions, non_empty_test_data["ipa"])

        empty_test_data_predictions = clean_predictions_batch(
            pipe(empty_test_data["audio"]), is_remove_space=is_remove_space
        )
        phone_lengths = model_eval_tracker.eval_empty_transcriptions(model, empty_test_data_predictions)

        # Write detailed by example evaluation if desired
        if verbose_results_dir:
            verbose_results_dir.mkdir(parents=True, exist_ok=True)
            # Take file separators out of model name
            clean_model_name = str(model).replace("/", "_").replace("\\", "_")
            hallucinations_csv = verbose_results_dir / (f"{clean_model_name}_{HALLUCINATIONS_SUFFIX}")
            detailed_results_csv = verbose_results_dir / (f"{clean_model_name}_{DETAILED_PREDICTIONS_CSV_SUFFIX}")

            empty_test_to_write = (
                empty_test_data.add_column("prediction", empty_test_data_predictions)
                .add_column("num_hallucinated_phones", phone_lengths)
                .remove_columns(["audio"])
            )
            empty_test_to_write.to_csv(hallucinations_csv, index=False)

            detailed_results = non_empty_test_data.add_column("prediction", predictions).remove_columns(["audio"])
            for k in ["phone_error_rates", "phone_feature_error_rates", "feature_error_rates"]:
                detailed_results = detailed_results.add_column(k, metrics[k])

            if "__index_level_0__" in detailed_results.column_names:
                detailed_results = detailed_results.remove_columns(["__index_level_0__"])

            detailed_results.to_csv(detailed_results_csv, index=False)

    # Write final metrics results for all models
    model_eval_tracker.to_csv(eval_csv)


def main_cli():
    parser = argparse.ArgumentParser(description="Evaluate audio to IPA transcriptions on a held out test corpus.")
    parser.add_argument(
        "-l", "--local_models", type=Path, nargs="*", help="List of paths to model files saved locally to evaluate."
    )

    parser.add_argument(
        "-f",
        "--hf_models",
        type=str,
        nargs="*",
        help="List of names of models stored on Hugging Face to download and evaluate.",
    )

    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        required=True,
        help="Path local Buckeye format pre-processed dataset to use for testing.",
    )

    parser.add_argument(
        "-e", "--eval_out", type=Path, required=True, help="Output CSV files to write evaluation metrics to."
    )
    parser.add_argument(
        "-v",
        "--verbose_results_dir",
        type=Path,
        help="Path to folder to dump all transcriptions to for later inspection.",
    )

    parser.add_argument(
        "-ns", "--no_space", action="store_true", help="Use this flag remove spaces in IPA transcription."
    )

    parser.add_argument(
        "-g",
        "--use_gpu",
        action="store_true",
        help="Use a GPU for inference if available. Otherwise the model runs on CPUs.",
    )

    args = parser.parse_args()

    buckeye_test = load_buckeye_split(args.data_dir, "test")
    main(
        buckeye_test,
        args.eval_out,
        args.local_models,
        args.hf_models,
        args.verbose_results_dir,
        args.no_space,
        args.use_gpu,
    )


if __name__ == "__main__":
    main_cli()
