"""Evaluates given models on test data, writing evaluation metrics in a summary CSV file.
Detailed results on the test data may also be written if desired.

Currently only Buckeye test split data is supported for evaluation.
"""

import argparse
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

import datasets
import evaluate
import ipatok
import kaldialign
import pandas as pd
import panphon.distance
import transformers
import torch

from multipa.data_utils import load_buckeye_split, clean_text, EMPTY_TRANSCRIPTION

PHONE_ERRORS_EVALUATOR = evaluate.load("ginic/phone_errors")
DETAILED_PREDICTIONS_CSV_SUFFIX = "detailed_predictions.csv"
HALLUCINATIONS_SUFFIX = "hallucinations.csv"

# Headers for detailed predictions CSV outputs
PREDICTION_KEY = "prediction"
PER_KEY = "phone_error_rates"
PFER_KEY = "phone_feature_error_rates"
FER_KEY = "feature_error_rates"
HALLUCITATIONS_KEY = "num_hallucinations"

# Null symbol for kaldi align that shouldn't ever occur in IPA strings
EPS = "***"


def clean_model_name(model_name: str | Path) -> str:
    """Removes path characters from models names or directories.

    Args:
        model_name: str, name of a model, usually from HuggingFace
    """
    return str(model_name).replace("/", "_").replace("\\", "_")


def compute_edit_distance_errors(prediction: str, reference: str, use_ipa_tokenise: bool = True, **kwargs):
    """Compares two strings and returns counts of the substitions, deletions and insertions
    between them.

    Results are in the following format:
    - Substitutions: dictionary mapping reference token to dict mapping substituted token to count
    - Deletions: dictionary mapping deleted tokens to count of number of times deleted
    - Insertions: dictionary mapping inserted token to number of times its inserted

    Args:
        prediction: _description_
        reference: _description_
        is_use_ipa_tokenise: _description_. Defaults to True.
        kwargs: any arguments to pass to the ipatok tokenise function

    Returns:
        tuple[Counter[tuple[str, str]], Counter[str], Counter[str]: Substitutions, deletions, insertions
    """
    if use_ipa_tokenise:
        pred_tokens = ipatok.tokenise(prediction, **kwargs)
        ref_tokens = ipatok.tokenise(reference, **kwargs)
    else:
        pred_tokens = list(prediction)
        ref_tokens = list(reference)

    subs = Counter()
    insertions = Counter()
    deletions = Counter()

    aligned_pairs = kaldialign.align(ref_tokens, pred_tokens, EPS)

    for r, p in aligned_pairs:
        if r == EPS:
            insertions[p] += 1
        elif p == EPS:
            deletions[r] += 1
        elif r != p:
            subs[(r, p)] += 1

    return subs, deletions, insertions


def collate_edit_distances(edit_dist_counter: list[Counter[Any]]) -> Counter[Any]:
    final_counts = Counter()
    for c in edit_dist_counter:
        final_counts += c

    return final_counts


class ModelEvaluator:
    model_key = "model"
    # Metric names that will become column headers
    per_key = "mean_phone_error_rate"
    pfer_key = "mean_phone_feature_error_rate"
    fer_key = "mean_feature_error_rate"
    phone_hallucinations_key = "total_phone_hallucinations"
    substitutions_key = "substitutions"
    deletions_key = "deletions"
    insertions_key = "insertions"

    def __init__(self, use_ipa_tokenise: bool = True):
        """Configure distance calculations and save results for each model

        Args:
            use_ipa_tokenise: Set as True to tokenise IPA phonemes before edit distance calculations or False
                to compute edit distance at the character level. Defaults to True.
        """
        self.distance_computer = panphon.distance.Distance()
        # Final results will have these keys
        # {model name -> {metric_key: metric_value}}
        self.results_to_write = defaultdict(dict)
        self.use_ipa_tokenise = use_ipa_tokenise

    def eval_edit_distances(self, model_name, predictions, references):
        # Compute errors for each example
        subs, deletions, inserts = [], [], []

        for p, r in zip(predictions, references):
            s, d, i = compute_edit_distance_errors(p, r, use_ipa_tokenise=self.use_ipa_tokenise)
            subs.append(s)
            deletions.append(d)
            inserts.append(i)

        # Save totals for the model
        total_subs = collate_edit_distances(subs)
        total_deletions = collate_edit_distances(deletions)
        total_inserts = collate_edit_distances(inserts)
        for k, curr_counts in [
            (ModelEvaluator.substitutions_key, total_subs),
            (ModelEvaluator.deletions_key, total_deletions),
            (ModelEvaluator.insertions_key, total_inserts),
        ]:
            if model_name in self.results_to_write and k in self.results_to_write[model_name]:
                prev_counts = self.results_to_write[model_name][k]
                self.results_to_write[model_name][k] = collate_edit_distances([curr_counts, prev_counts])
            else:
                self.results_to_write[model_name][k] = curr_counts

        # Return by example results
        edit_dist_dict = {
            ModelEvaluator.substitutions_key: subs,
            ModelEvaluator.deletions_key: deletions,
            ModelEvaluator.insertions_key: inserts,
        }
        return edit_dist_dict

    def eval_non_empty_transcriptions(self, model_name, predictions, references):
        """Compare the predictions and gold-standard references for a model,
        then add results to model results tracker.
        Returns the full detailed evaluation results as a dictionary.
        """
        metrics = PHONE_ERRORS_EVALUATOR.compute(predictions=predictions, references=references)
        for k in [ModelEvaluator.per_key, ModelEvaluator.pfer_key, ModelEvaluator.fer_key]:
            self.results_to_write[model_name][k] = metrics[k]

        edit_dist_dict = self.eval_edit_distances(model_name, predictions, references)
        metrics.update(edit_dist_dict)
        return metrics

    def eval_empty_transcriptions(self, model_name, predictions):
        """Count number of phone hallucinations for this model and save to write later.
        Returns the detailed evaluation results as a list with one entry per prediction.
        """
        # Insertions according to panphon's feature model
        phone_lengths = [len(self.distance_computer.fm.ipa_segs(p)) for p in predictions]
        total_phone_hallucinations = sum(phone_lengths)
        self.results_to_write[model_name][ModelEvaluator.phone_hallucinations_key] = total_phone_hallucinations

        # Insertions according to edit distance
        edit_dist_dict = self.eval_edit_distances(model_name, predictions, [""] * len(predictions))
        edit_dist_dict[HALLUCITATIONS_KEY] = phone_lengths
        return edit_dist_dict

    def to_csv(self, csv_path: Path | str):
        """Write the aggregate evaluation results stored in this object to the specified CSV file.
        Each model is a row, with aggregate (average or total) metrics stored per column
        """
        df = pd.DataFrame.from_dict(self.results_to_write, orient="index")
        df.index.name = ModelEvaluator.model_key
        df.to_csv(csv_path)

    def write_edit_distance_results(self, model_name: str | Path, directory: Path):
        """Writes counts of edit distance errors by symbol to CSV files.
        Each model will have 3 corresponding CSVs, one each for subsitutions, deletions and insertions.

        Args:
            model_name: The name of the model being evaluated
            directory: The desired output directory where CSV files will be written
        """
        csv_base_name = clean_model_name(model_name)
        for k in [ModelEvaluator.deletions_key, ModelEvaluator.insertions_key]:
            count_col = f"total_{k}"
            edit_dist_df = pd.DataFrame.from_records(
                list(self.results_to_write[model_name][k].items()), columns=["symbol", count_col]
            )
            edit_dist_df.sort_values(by=count_col, ascending=False, inplace=True)
            edit_dist_df.to_csv(directory / f"{csv_base_name}_{k}.csv", index=False)

        subs_col = f"total_{ModelEvaluator.substitutions_key}"
        substitutions_df = pd.DataFrame.from_records(
            [(s[0], s[1], v) for s, v in self.results_to_write[model_name][ModelEvaluator.substitutions_key].items()],
            columns=["original", "substitution", subs_col],
        )
        substitutions_df.sort_values(by=subs_col, inplace=True, ascending=False)
        substitutions_df.to_csv(directory / f"{csv_base_name}_{ModelEvaluator.substitutions_key}.csv", index=False)


def preprocess_test_data(test_dataset: datasets.Dataset, is_remove_space: bool = False, num_proc: int | None = None):
    """
    Filters the test dataset into examples with non-empty and empty transcriptions,
    since they should be evaluated separately.
    Also performs additional text cleaning if specified.

    Args:
        test_dataset: Huggingface dataset you'll use for evaluation
        is_remove_space: Filter out spaces in IPA strings if true
        num_proc: The number of processes to use for multiprocessing. If None, no multiprocessing is used.

    Returns:
        non_empty_transcriptions_dataset, empty_transcriptions_dataset: a tuple of Huggingface datasets
    """
    # Set sampling rate to 16K
    input_data = test_dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000)).map(
        lambda x: clean_text(x, is_remove_space=is_remove_space), num_proc=num_proc
    )

    empty_test_data = input_data.filter(lambda x: x["ipa"] == EMPTY_TRANSCRIPTION, num_proc=num_proc)
    non_empty_test_data = input_data.filter(lambda x: x["ipa"] != EMPTY_TRANSCRIPTION, num_proc=num_proc)

    return non_empty_test_data, empty_test_data


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
    eval_csv: Path | str,
    local_models: list[Path] | None = None,
    hf_models: list[str] | None = None,
    verbose_results_dir: Path | None = None,
    is_remove_space: bool = False,
    use_gpu: bool = False,
    num_proc: int | None = None,
    edit_dist_dir: Path | None = None,
):
    if local_models is None:
        local_models = []
    if hf_models is None:
        hf_models = []

    print("Loading test data")
    non_empty_test_data, empty_test_data = preprocess_test_data(input_data, is_remove_space, num_proc)
    print("Number of test examples with NON-empty transcriptions:", len(non_empty_test_data))
    print(non_empty_test_data)

    print("Number of test examples with empty transcriptions:", len(empty_test_data))
    print(empty_test_data)

    model_eval_tracker = ModelEvaluator()

    selected_torch_device = get_torch_device(use_gpu)

    for model in local_models + hf_models:
        print("Evaluating model:", model)
        pipe = transformers.pipeline("automatic-speech-recognition", model=model, device=selected_torch_device)

        if len(non_empty_test_data) > 0:
            print("Getting predictions for audio with non-empty gold-standard transcriptions")
            predictions = datasets.Dataset.from_list(pipe(non_empty_test_data["audio"]))
            predictions = predictions.map(
                lambda x: clean_text(x, text_key="text", is_remove_space=is_remove_space), num_proc=num_proc
            )
            predictions = predictions.rename_column("text", PREDICTION_KEY)
            print("Predictions data preview:")
            print(predictions[0])

            print("Computing performance metrics for non-empty audio transcriptions")
            metrics = model_eval_tracker.eval_non_empty_transcriptions(
                model, predictions[PREDICTION_KEY], non_empty_test_data["ipa"]
            )

        if len(empty_test_data) > 0:
            print("Getting predictions for audio with empty gold-standard transcriptions")
            empty_test_data_predictions = datasets.Dataset.from_list(pipe(empty_test_data["audio"]))
            empty_test_data_predictions = empty_test_data_predictions.map(
                lambda x: clean_text(x, text_key="text", is_remove_space=is_remove_space), num_proc=num_proc
            )
            empty_test_data_predictions = empty_test_data_predictions.rename_column("text", PREDICTION_KEY)
            phone_lengths = model_eval_tracker.eval_empty_transcriptions(
                model, empty_test_data_predictions[PREDICTION_KEY]
            )

        # Write by-model edit distance errors with counts
        if edit_dist_dir:
            print("Writing edit distance results to", edit_dist_dir)
            edit_dist_dir.mkdir(parents=True, exist_ok=True)
            model_eval_tracker.write_edit_distance_results(model, edit_dist_dir)

        # Write detailed by example evaluation if desired
        if verbose_results_dir:
            print("Writing detailed transcription predictions and metrics to", verbose_results_dir)
            clean_model_id = clean_model_name(model)
            verbose_results_dir.mkdir(parents=True, exist_ok=True)
            # Take file separators out of model name
            hallucinations_csv = verbose_results_dir / (f"{clean_model_id}_{HALLUCINATIONS_SUFFIX}")
            detailed_results_csv = verbose_results_dir / (f"{clean_model_id}_{DETAILED_PREDICTIONS_CSV_SUFFIX}")

            if len(empty_test_data) > 0:
                empty_test_to_write = empty_test_data_predictions.add_column(
                    "num_hallucinated_phones", phone_lengths
                ).remove_columns(["audio"])
                empty_test_to_write.to_csv(hallucinations_csv, index=False)

            if len(non_empty_test_data) > 0:
                detailed_results = non_empty_test_data.add_column(
                    PREDICTION_KEY, predictions[PREDICTION_KEY]
                ).remove_columns(["audio"])
                for k in ["phone_error_rates", "phone_feature_error_rates", "feature_error_rates"]:
                    detailed_results = detailed_results.add_column(k, metrics[k])

                if "__index_level_0__" in detailed_results.column_names:
                    detailed_results = detailed_results.remove_columns(["__index_level_0__"])

                detailed_results.to_csv(detailed_results_csv, index=False)

    # Write final aggregate metrics results for all models
    eval_csv.parent.mkdir(exist_ok=True, parents=True)
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
        help="If desired, specify a path to a directory to dump by-model transcriptions with by-example performance metrics to for later inspection if desired. For each model, one CSV for detailed predictions and one for hallucinations (audio without speech, but where transcripts) are created.",
    )

    parser.add_argument(
        "-ed",
        "--edit_dist_dir",
        type=Path,
        help="If desired, specify a path to a directory for storing by-model edit distance results, 3 CSVs for each model: one for substitions, insertions and deletions",
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

    parser.add_argument(
        "--num_proc",
        type=int,
        help="Specify the number of CPUs for preprocessing. If unset, no multiprocessing is used.",
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
        args.num_proc,
        args.edit_dist_dir,
    )


if __name__ == "__main__":
    main_cli()
