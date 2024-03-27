import argparse
from pathlib import Path

import datasets
import evaluate
import pandas as pd

import transformers

from multipa.data_utils import load_buckeye_split, clean_text




def main(input_data:datasets.Dataset, eval_csv, local_models:list[Path]|None=None, hf_models:list[str]|None=None, 
         verbose_results_dir:Path|None=None, is_remove_space:bool=False):
    # Set sampling rate to 16K

    test_dataset = input_data.cast_column("audio", datasets.Audio(sampling_rate=16_000)).\
        map(lambda x: clean_text(x, is_remove_space=is_remove_space))
    
    #empty_reference_test_data = test_data.filter
    #non_empty_reference_test_data
    phone_errors = evaluate.load("ginic/phone_errors")
    results = {}
    for model in local_models + hf_models: 
        processor = transformers.AutoProcessor.from_pretrained(model)
        pipe = transformers.pipeline("automatic-speech-recognition", model=model)
        predictions = processor(test_dataset["audio"])



        phone_errors.compute()






def main_cli():
    parser = argparse.ArgumentParser(description = "Evaluate audio to IPA transcriptions on a held out test corpus.")
    parser.add_argument("-l", "--local_models", type=Path, nargs="*", 
                        help="List of paths to model files saved locally to evaluate.")
    
    parser.add_argument("-h", "--hf_models", type=str, nargs="*", 
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