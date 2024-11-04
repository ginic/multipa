from pathlib import Path
from typing import Union

import datasets

# Constant corpus identifier options
LIBRISPEECH_KEY = "librispeech"
COMMONVOICE_KEY = "commonvoice"
BUCKEYE_KEY = "buckeye"

# Extra vocabulary elements for tokenization
UNKNOWN_TOKEN = "[UNK]"
PADDING_TOKEN = "[PAD]"
EMPTY_TRANSCRIPTION = ""

class DataLoadError(Exception):
    pass


def filter_low_quality(dataset):
    dataset = dataset.filter(lambda batch: batch["down_votes"] == 0)
    return dataset


def remove_space(batch: dict, col_key) -> dict:
    ipa = batch[col_key]
    ipa = ipa.split()
    ipa = "".join(ipa)
    batch[col_key] = ipa
    return batch


def replace_none(batch: dict, col_key) -> dict:
    """Replaces any None value for the given col_key with the empty string '' instead. 

    Args:
        batch (dict): Dictionary containing specified col)key

    Returns:
        dict: 
    """
    ipa = batch[col_key]
    if ipa is None:
        batch[col_key] = EMPTY_TRANSCRIPTION    
    return batch 

def clean_text(batch:dict, text_key="ipa", is_remove_space=True):
    """Basic text pre-processing steps. Replace any None values with the empty string and optionally remove whitespace.

    Args:
        batch (dict): attributes for a dataset sample
        text_key (str, optional): Column/dict key where the desired text is stored. Defaults to "ipa".
        is_remove_space (bool, optional): Set to true to remove whitespace from text. Defaults to True.

    Returns:
        dict: batch with clean text replacing original text_key value
    """
    batch = replace_none(batch, text_key)
    if is_remove_space:
        batch = remove_space(batch, text_key)
    return batch

def validate_dataset_files_match(raw_data, ipa_data, key:str, is_check_basename:bool=False):
    if len(raw_data) != len(ipa_data):
        raise DataLoadError("Length of raw data and IPA transcription data doesn't match.")

    for j in range(len(raw_data)):
        if is_check_basename:
            filename = raw_data[j][key].split("/")[-1]
            ipa_filename = ipa_data[j][key].split("/")[-1]
        else:
            filename = raw_data[j][key]
            ipa_filename = ipa_data[j][key]
        if filename != ipa_filename:
            raise DataLoadError(f"No match between IPA and raw data on '{key}'. IPA: {ipa_filename}, Raw: {filename}")
        
def join_column(left_dataset, right_dataset, on_key:str, right_col:str, is_check_basename:bool=False, 
                additional_check_col:Union[str, None]="sentence"):
    """Joins a column from the right dataset into the left.

    Args:
        left_dataset: Huggingface dataset 
        right_dataset: Huggingface dataset
        on_key (str): join key, must be present in both datasets
        right_col (str): column to add from right dataset
        is_check_basename (bool): If on_key contains full paths, but you only want to validate the basename matches
        additional_check_col (str|None): If there's an additional column you want to check matches before joining, use this. Check 'sentence' column by default.

    Returns:
        Huggingface Dataset
    """
    # TODO Better to use SQL style join on file column if Huggingface supports this?
    # Join in IPA data by matching file name
    right_sorted = right_dataset.sort(on_key)
    left_sorted = left_dataset.sort(on_key)
    validate_dataset_files_match(left_sorted, right_sorted, on_key, is_check_basename)
    if additional_check_col is not None: 
        validate_dataset_files_match(left_sorted, right_sorted, additional_check_col)
    new_col = [right_sorted[i][right_col] for i in range(len(right_sorted))]
    return left_sorted.add_column(right_col, new_col)


def load_common_voice_split(language: str, quality_filter: bool, split:str, huggingface_split:str, data_dir:str, json_filename:str, 
                            cache_dir:str, num_proc:int, dataset_name:str="mozilla-foundation/common_voice_11_0"):
    """Loads the specified split of Common Voice dataset, reading IPA transcriptions for a local JSON file. 
    Optionally filter to only include high quality data from Common Voice. 
    Always does special language specific filtering for Tamil, language="ta"

    Args:
        language (str): 2-letter language ISO code
        quality_filter (bool): Set to true to remove low quality transcriptions with at least one downvote
        split (str): Data split name to read from the JSON file
        huggingface_split (str): Data split name for loading directly from Huggingface
        data_dir (str): Path to directory containing the json dataset
        json_filename (str): Json filename (basename only). Should be in data_dir.
        cache_dir (str): Cache directory for Huggingface
        num_proc (int): number of threads when loading dataset from Huggingface
        dataset_name (str, optional): _description_. Defaults to "mozilla-foundation/common_voice_11_0".

    Returns:
        Huggingface Dataset
    """
    ipa_dataset = datasets.load_dataset("json",
                                data_files=str(Path(data_dir) / json_filename),
                                split=split)
    raw_audio = datasets.load_dataset(dataset_name,
                             language,
                             split=huggingface_split,
                             num_proc=num_proc, 
                             cache_dir=cache_dir)
    
    full_dataset = join_column(raw_audio, ipa_dataset, "path", "ipa", is_check_basename=True)

    # Remove Tamil sentences containing "ச"
    if language == "ta":
        full_dataset = full_dataset.filter(lambda batch: "ச" not in batch["sentence"]) 

    if quality_filter:
        full_dataset = filter_low_quality(full_dataset)
    
    return full_dataset                                


def load_librispeech_split(split:str, huggingface_split:str, data_dir:str, json_filename:str, cache_dir:str, 
                           num_proc:int, dataset_name:str = "librispeech_asr"):
    """Load a full split of the Librispeech dataset from Huggingface, reading IPA transcriptions from a local JSON file. 
    Rename columns to match Common Voice format, so training and validation code can be shared.

    Args:
        split (str): Name fo the data split for reading IPA data from json dataset format - should be either "train" or "valid"
        huggingface_split (str): Name of the data split to download from Huggingface. 
        data_dir (str): Path to directory containing the json dataset
        json_filename (str): Json filename (basename only). Should be in data_dir.
        cache_dir (str): Cache directory for Huggingface
        num_proc (int): number of threads when loading dataset from Huggingface
        dataset_name (str): Name of the dataset to load from Huggingface, defaults to "librispeech_asr"
    """
    # Librispeech starts with the audio path in "file" column and transcription in "text" column
    # You need to finish with audio path in "path" and transcription "sentence"
    ipa_dataset = datasets.load_dataset("json",
                                data_files=str(Path(data_dir) / json_filename),
                                split=split)
    ipa_dataset = ipa_dataset.rename_column("text", "sentence")

    raw_audio = datasets.load_dataset(dataset_name,
                             split=huggingface_split,
                             num_proc=num_proc, 
                             cache_dir=cache_dir)
    raw_audio = raw_audio.rename_column("text", "sentence")

    # Join in IPA data by matching file name
    full_dataset = join_column(raw_audio, ipa_dataset, "file", "ipa")
    full_dataset = full_dataset.rename("file", "path")
    return full_dataset

def load_buckeye_split(corpus_root_dir: str, split:str):
    dataset_split = datasets.load_dataset("audiofolder", data_dir=corpus_root_dir, split=split)
    # Output data has duplicates despite following the format from HuggingFace documentation, so deduplicate based on utterance id
    deduplicated_df = dataset_split.to_pandas().drop_duplicates("utterance_id")
    return datasets.Dataset.from_pandas(deduplicated_df)

    
