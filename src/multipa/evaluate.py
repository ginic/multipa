"""Evaluates given models on test data, writing evaluation metrics in a summary CSV file. 
Detailed results on the test data may also be written if desired.
"""
import argparse
from pathlib import Path

import datasets
import evaluate

import pandas as pd
import transformers
import panphon.distance

from multipa.data_utils import load_buckeye_split, clean_text, EMPTY_TRANSCRIPTION
 

def main(input_data:datasets.Dataset, eval_csv, local_models:list[Path]|None=None, hf_models:list[str]|None=None, 
         verbose_results_dir:Path|None=None, is_remove_space:bool=False):
    if local_models is None:
        local_models = []
    if hf_models is None:
        hf_models = []

    # Set sampling rate to 16K
    test_dataset = input_data.cast_column("audio", datasets.Audio(sampling_rate=16_000)).\
        map(lambda x: clean_text(x, is_remove_space=is_remove_space))
    
    empty_test_data = test_dataset.filter(lambda x: x["ipa"] == EMPTY_TRANSCRIPTION)
    print("Number of test examples with empty transcriptions:", len(empty_test_data))
    print(empty_test_data)
    non_empty_test_data = test_dataset.filter(lambda x: x["ipa"] != EMPTY_TRANSCRIPTION)
    phone_errors = evaluate.load("ginic/phone_errors") 

    # Final results will have these keys
    model_key = "model"
    per_key = "mean_phone_error_rate"
    pfer_key = "mean_phone_feature_error_rate"
    fer_key = "mean_feature_error_rate"
    phone_hallucinations_key = "total_phone_hallucinations"
    results_to_write = {model_key:[], per_key:[], pfer_key:[], fer_key:[], phone_hallucinations_key:[]}
    
    for model in local_models + hf_models: 
        print("Evaluating model:", model)
        pipe = transformers.pipeline("automatic-speech-recognition", model=model)
        predictions = [d["text"] for d in pipe(non_empty_test_data["audio"])]
        metrics = phone_errors.compute(predictions=predictions, references=non_empty_test_data["ipa"])
        
        empty_test_data_predictions = [d["text"] for d in pipe(empty_test_data["audio"])]
        distance_computer = panphon.distance.Distance()
        phone_lengths = [len(distance_computer.fm.ipa_segs(p)) for p in empty_test_data_predictions]
        total_phone_hallucinations = sum(phone_lengths)

        # Collect results for this model
        results_to_write[model_key].append(model)
        results_to_write[phone_hallucinations_key].append(total_phone_hallucinations)
        for k in [per_key, pfer_key, fer_key]:
            results_to_write[k].append(metrics[k])
        
        # Write detailed by example evaluation if desired
        if verbose_results_dir:
            verbose_results_dir.mkdir(parents=True, exist_ok=True)
            # Take file separators out of model name
            clean_model_name = str(model).replace("/", "_").replace("\\", "_")
            hallucinations_csv =  verbose_results_dir / (clean_model_name + "_hallucinations.csv")
            detailed_results_csv = verbose_results_dir / (clean_model_name + "_detailed_predictions.csv")

            empty_test_to_write = empty_test_data.add_column("prediction", empty_test_data_predictions).\
                add_column("num_hallucinated_phones", phone_lengths).\
                remove_columns(["audio"])
            empty_test_to_write.to_csv(hallucinations_csv, index=False)

            detailed_results = non_empty_test_data.add_column("prediction", predictions).\
                remove_columns(["audio"])
            for k in ["phone_error_rates", "phone_feature_error_rates", "feature_error_rates"]:
                detailed_results = detailed_results.add_column(k, metrics[k])
            detailed_results.to_csv(detailed_results_csv, index=False)            

    # Write final metrics results
    pd.DataFrame(results_to_write).to_csv(eval_csv, index=False)


def main_cli():
    parser = argparse.ArgumentParser(description = "Evaluate audio to IPA transcriptions on a held out test corpus.")
    parser.add_argument("-l", "--local_models", type=Path, nargs="*", 
                        help="List of paths to model files saved locally to evaluate.")
    
    parser.add_argument("-f", "--hf_models", type=str, nargs="*", 
                        help="List of names of models stored on Hugging Face to download and evaluate.")
    
    parser.add_argument("-d", "--data_dir", type=Path, required=True, 
                        help="Path local Buckeye pre-processed dataset to use for testing.")

    parser.add_argument("-e", "--eval_out", type=Path, required=True,
                        help="Output CSV files to write evaluation metrics to.")
    parser.add_argument("-v", "--verbose_results_dir", type=Path,
                        help="Path to folder to dump all transcriptions to for later inspection.")  

    parser.add_argument("-ns", "--no_space", action='store_true',
                        help="Use this flag remove spaces in IPA transcription.") 
    
    args = parser.parse_args()
    
    buckeye_test = load_buckeye_split(args.data_dir, "test")
    main(buckeye_test, args.eval_out, args.local_models, args.hf_models, 
         args.verbose_results_dir, args.no_space)
    
    
if __name__ == "__main__":
    main_cli()