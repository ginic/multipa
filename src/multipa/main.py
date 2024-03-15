import argparse
import gc
import importlib.resources
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from dataclasses import dataclass
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import torch

from multipa.data_utils import filter_low_quality, BUCKEYE_KEY, COMMONVOICE_KEY, LIBRISPEECH_KEY, DataLoadError

# Extra vocabulary elements for tokenization
UNKNOWN_TOKEN = "[UNK]"
PADDING_TOKEN = "[PAD]"

#from multipa.add_forvo import add_language

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        # and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
                )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def extract_all_chars_ipa(batch: dict) -> dict:
    # Change this function later at some point to create vocabulary based on
    # phonemes, not on characters
    all_text = " ".join(batch["ipa"])
    return set(all_text)

def prepare_dataset_ipa(batch: dict, processor_ipa:Wav2Vec2Processor) -> dict:
    audio = batch["audio"]

    # batched output is unbatched
    batch["input_values"] = processor_ipa(audio["array"],
                                          sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor_ipa.as_target_processor():
        batch["labels"] = processor_ipa(batch["ipa"]).input_ids
    return batch

def remove_long_data(dataset, max_seconds=12):
    # convert pyarrow table to pandas
    dftest = dataset.to_pandas()
    # find out length of input_values
    dftest['len'] = dftest['input_values'].apply(len)
    # for wav2vec training we already resampled to 16khz
    # remove data that is longer than max_seconds (6 seconds ideal)
    maxLength = max_seconds * 16000 
    dftest = dftest[dftest['len'] < maxLength]
    dftest = dftest.drop('len', 1)
    # convert back to pyarrow table to use in trainer
    dataset = dataset.from_pandas(dftest)
    # directly remove do not wait for gc
    del dftest
    return dataset

def concatenate_common_voice(datasetlist: list):
    """
    Concatenate more than one datasets from Common Voice.
    Also consider using datasets.interleave_datasets(datasets: List[DatasetType]
    so that the new dataset is constructed by cycling between each source to get the examples.
    """
    init_data = datasetlist[0]
    for d in datasetlist:
        assert d.features.type == init_data.features.type
    concatenated = concatenate_datasets(datasetlist)
    return concatenated

def remove_space(batch: dict) -> dict:
    ipa = batch["ipa"]
    ipa = ipa.split()
    ipa = "".join(ipa)
    batch["ipa"] = ipa
    return batch

def create_vocabulary(*datasets):
    """Determines the vocabulary of IPA characters needed for the model.

    Returns:
        dict: vocab -> index
    """
    vocab_set = set()
    for d in datasets:
        d_vocab = d.map(
            extract_all_chars_ipa,
            batched=True,
            batch_size=-1,
            keep_in_memory=True,
            remove_columns=d.column_names
        )
        vocab_set = vocab_set | d_vocab

    # Add in data from resources file
    all_vocab_file = importlib.resources.files("multipa.resources").joinpath("full_vocab_ipa.txt")
    with importlib.resources.as_file(all_vocab_file) as f:
        new_vocab = set([l.strip() for l in f.read_text().splitlines()])
        vocab_set = vocab_set | new_vocab
    
    vocab_dict_ipa = {v: k for k, v in enumerate(vocab_set)}
    vocab_dict_ipa[UNKNOWN_TOKEN] = len(vocab_dict_ipa)
    vocab_dict_ipa[PADDING_TOKEN] = len(vocab_dict_ipa)
    return vocab_dict_ipa


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
                additional_check_col:str|None="sentence"):
    """Joins a column from the right dataset into the left.

    Args:
        left_dataset: Huggingface dataset 
        right_dataset: Huggingface dataset
        on_key (str): join key, must be present in both datasets
        right_col (str): column to add from right dataset
        is_check_basename (bool): If on_key contains full paths, but you only want to validate the basename matches
        additional_check_col (str|None): If there's an additional column you want to check matches before joining, use this. Check 'sentence' column by default.
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
        _type_: _description_
    """
    ipa_dataset = load_dataset("json",
                                data_files=str(Path(data_dir) / json_filename),
                                split=split)
    raw_audio = load_dataset(dataset_name,
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
    ipa_dataset = load_dataset("json",
                                data_files=str(Path(data_dir) / json_filename),
                                split=split)
    ipa_dataset = ipa_dataset.rename_column("text", "sentence")

    raw_audio = load_dataset(dataset_name,
                             split=huggingface_split,
                             num_proc=num_proc, 
                             cache_dir=cache_dir)
    raw_audio = raw_audio.rename_column("text", "sentence")

    # Join in IPA data by matching file name
    full_dataset = join_column(raw_audio, ipa_dataset, "file", "ipa")
    full_dataset = full_dataset.rename("file", "path")
    return full_dataset
    

def main_cli():
    # Arguments
    parser = argparse.ArgumentParser(description="Trains the speech recognition model. Specify corpus, "\
                                     "model training parameters and language details if needed. ")
    parser.add_argument("-e", "--num_train_epochs", type=int, default=30,
                        help="Specify the number of train epochs. By default it's set to 30.")
    parser.add_argument("--num_proc", type=int, default=8,
                        help="Specify the number of CPUs for preprocessing. Default set to 24.")

    parser.add_argument("-ml", "--max-length", type=int, default=12, help="Maximum audio length of training & validation samples in seconds")
    parser.add_argument("-ns", "--no_space", type=bool, default=False,
                        help="Set True if you want to remove spaces from the training and test data.") 
    parser.add_argument("-o", "--output_dir", type=str,
                        help="Specify the directory to save files for vocab, stats and trained models.")

    
    # TODO This is a bit confusing, but it's basically reading the train/test splits from the preprocessing output. Might not be necessary for Buckeye
    parser.add_argument("-dd", "--data_dir", type=str, default="data_new",
                        help="Specify the directory path for the training/validation data files." \
                        "Default is set to `data_new/`, which stores the data from the as-of-now newest" \
                        "`mozilla-foundation/common_voice_11_0`.")
    
    # TODO Can become subparser
    #parser.add_argument("-ds", "--dataset", type=str, default="mozilla-foundation/common_voice_11_0",
    #                    help="Specify the dataset name. Default is set to" \
    #                    "`mozilla-foundation/common_voice_11_0`.")
    
    # TODO Unclear if needed for buckeye
    parser.add_argument("--cache_dir", type=str, default="~/.cache/huggingface/datasets",
                        help="Specify the cache directory's path if you choose to load dataset from non-default cache.")
    # Ignore Forvo data for now
    # parser.add_argument("-a", "--additional_data", nargs=1, type=bool, default=False, 
    #                    help="Specify if you want to use additional data fetched from Forvo.")

    subparsers = parser.add_subparsers(help="Specify which corpus you'll be using", dest="corpus")

    comm_voice_subparser = subparsers.add_parser(COMMONVOICE_KEY, help="Use the Common Voice corpus version 11 from the Huggingface data repo.")

    comm_voice_subparser.add_argument("-l", "--languages", nargs="+", type=str, required=True,
                        help="Specify language code (split by space). Typically ISO639-1, or ISO639-2 if not found in ISO639-1.")
    
    # TODO Needs to be a list only for Common Voice, for Buckeye and Librispeech can be just one value
    comm_voice_subparser.add_argument("-tr", "--train_samples", nargs="+", type=int,
                        help="Specify the number of samples to be used as the training data for each language. " \
                        "For example, if you want to use 1000, 2000, 3000 training samples for Japanese, Polish, " \
                        "and Maltese, then specify as -l ja pl mt -tr 1000 2000 3000." \
                        "You can type an irrationally large number to pick up the maximum value.")
    
    # TODO Needs to be a list only for Common Voice, for Buckeye and Librispeech can be just one value
    comm_voice_subparser.add_argument("-ve", "--val_samples", nargs="+", type=int,
                        help="Specify the number of samples to be used as the test data for each language. " \
                        "For example, if you want to use 1000, 2000, 3000 test samples for Japanese, Polish, " \
                        "and Maltese, then specify as -l ja pl mt -tr 1000 2000 3000. " \
                        "You can type an irrationally large number to pick up the maximum value.")

    comm_voice_subparser.add_argument("-qf", "--quality_filter", type=bool, default=True,
                        help="Specify if you want to remove low quality audio (at least having 1 down vote) from the dataset." \
                        "True if you want to, False if you do not want to.")
    
    librispeech_subparser = subparsers.add_parser(LIBRISPEECH_KEY, help="Use the Librispeech ASR English corpus from the Huggingface data repo.")
    librispeech_subparser.add_argument("-tr", "--train_samples", type=int,
                        help="Specify the number of samples to be used as the training data. "\
                        "You can type an irrationally large number to pick up the maximum value.")
    librispeech_subparser.add_argument("-ve", "--val_samples", type=int,
                        help="Specify the number of samples to be used as the test data. "\
                        "You can type an irrationally large number to pick up the maximum value.")

    buckeye_subparser = subparsers.add_parser(BUCKEYE_KEY, help="Use the Buckeye corpus from a local Huggingface 'audiofolder'.")
    buckeye_subparser.add_argument("-tr", "--train_samples", type=int,
                        help="Specify the number of samples to be used as the training data. "\
                        "You can type an irrationally large number to pick up the maximum value.")
    buckeye_subparser.add_argument("-ve", "--val_samples", type=int,
                        help="Specify the number of samples to be used as the test data. "\
                        "You can type an irrationally large number to pick up the maximum value.")
        
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True)
        
    # Set up corpus stats tracking file
    stats_file = output_dir / "stats_train_valid.txt"
    with open(stats_file, "w") as f:
        f.write("corpus lang train valid\n")

    if args.corpus == LIBRISPEECH_KEY:
        dataset_name = "librispeech_asr"
        train_data = load_librispeech_split("train", "train.clean.100", args.data_dir, "en_train.json", 
                                            args.cache_dir, args.num_proc, dataset_name)
        valid_data = load_librispeech_split("valid", "validation.clean", args.data_dir, "en_valid.json")
        # Clipping to the specified sample size using datasets's Dataset.select()
        train_limit = min(args.train_samples, len(train_data))
        valid_limit = min(args.val_samples, len(valid_data))
        full_train_data = train_data.select(range(train_limit))
        full_valid_data = valid_data.select(range(valid_limit))

        with open(stats_file, "a") as f:
            f.write(f"{dataset_name} en {len(train_data)} {len(valid_data)}\n")

    elif args.corpus == BUCKEYE_KEY: 
        # TODO
        pass
    
    elif args.corpus == COMMONVOICE_KEY:
        lgx = args.languages   
        assert len(args.train_samples) <= len(lgx), "`train_samples` argument is longer than the number of languages"
        assert len(args.val_samples) <= len(lgx), "`val_samples` argument is longer than the number of languages"
     
        train_list = []
        valid_list = []
        dataset_name = "mozilla-foundation/common_voice_11_0"
        for i, lang in enumerate(lgx):
            train_sample = args.train_samples[i]
            valid_sample = args.val_samples[i]
            train_data = load_common_voice_split(lang, args.quality_filter, "train", "train", 
                                                 args.data_dir, f"{lang}_train.json", args.cache_dir, args.num_proc, dataset_name)
            valid_data = load_common_voice_split(lang, args.quality_filter, "valid", "validation", 
                                                 args.data_dir, f"{lang}_valid.json", args.cache_dir, args.num_proc, dataset_name)


            # Clipping to the specified sample size using datasets's Dataset.select()
            train_limit = min(train_sample, len(train_data))
            valid_limit = min(valid_sample, len(valid_data))
            train_data = train_data.select(range(train_limit))
            valid_data = valid_data.select(range(valid_limit))
            
            train_list.append(train_data)
            valid_list.append(valid_data)

            with open(stats_file, "a") as f:
                f.write(f"{dataset_name} {lang} {len(train_data)} {len(valid_data)}\n")
        
        # Concatenate the languages
        print("Concatenating datasets for each language...")
        full_train_data = concatenate_common_voice(train_list)
        full_valid_data = concatenate_common_voice(valid_list)
        print("Concatenation done")

    # This doesn't work right now and we don't have access to the Forvo data, so I'm commenting out
    # if args.additional_data:
    #     print("Concatenating the additional data from Forvo...")
    #     new_ds = add_language() # -> dict
    #     new_ds = new_ds.train_test_split(test_size=0.2)
    #     common_voice_train = concatenate_datasets([common_voice_train, new_ds["train"]])
    #     common_voice_valid = concatenate_datasets([common_voice_valid, new_ds["test"]])
    #     print("Concatenated additional data from Forvo")

    # Remove unnecessary columns
    unnecessary_columns = [
        "accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes", # for Common Voice
        "speaker_id", "chapter_id", "id" #for librispeech
        ]
    columns_to_remove = set(unnecessary_columns).intersection(full_train_data.column_names)
    print("Removing unnecessary columns:", columns_to_remove)
    full_train_data = full_train_data.remove_columns(columns_to_remove)
    full_valid_data = full_valid_data.remove_columns(columns_to_remove)
    print("Unnecessary columns removed. Data preview:")
    print(full_train_data[0])
    assert full_train_data.features.type == full_valid_data.features.type

    # Remove spaces if specified
    if args.no_space:
        full_train_data = full_train_data.map(remove_space)
        full_valid_data = full_valid_data.map(remove_space)
        assert " " not in full_train_data[0]["ipa"], print("Apparently space removal did not work correctly")
        
    # Shuffle the dataset
    print("Shuffling the dataset...")
    full_train_data = full_train_data.shuffle(seed=42)
    full_valid_data = full_valid_data.shuffle(seed=35)
    print("Shuffling done")

    # Preprocessing 
    print("Creating vocabulary...")
    vocab_dict_ipa = create_vocabulary(full_train_data, full_valid_data)

    print("Writing vocab json files...")
    # Don't forget to change the file name when you use different languages,
    # otherwise the vocab file will be lost
    vocab_file = output_dir / f"{args.corpus}_ipa_vocab.json"
    with open(vocab_file, 'w') as vocab_file_ipa:
        json.dump(vocab_dict_ipa, vocab_file_ipa)
    print("Vocab json files created")

    # Create Tokenizers
    print("Creating Tokenizers...")
    # Be careful to load the correct vocab file.
    tokenizer_ipa = Wav2Vec2CTCTokenizer(vocab_file,
                                         unk_token=UNKNOWN_TOKEN,
                                         pad_token=PADDING_TOKEN,
                                         word_delimiter_token="|")
    print("Tokenizers created") 

    # Create a Feature Extractor
    print("Creating Feature Extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                 sampling_rate=16_000,
                                                 padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=True)
    print("Feature Extractor created") 

    # Define Processors
    print("creating Processors...")
    processor_ipa = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                      tokenizer=tokenizer_ipa)
    print("Processors created") 

    # Set the sampling rate to 16,000Hz
    print("Adjusting the sampling rate to 16,000Hz...")
    full_train_data = full_train_data.cast_column("audio", Audio(sampling_rate=16_000))
    full_valid_data = full_valid_data.cast_column("audio", Audio(sampling_rate=16_000))
    print("Sampling rate adjustment done")

    print("Preprocessing the dataset...")
    # Try removing `num_proc=` if you encounter any errors while running this part
    processor_func = lambda x: prepare_dataset_ipa(x, processor_ipa)
    full_train_data = full_train_data.map(
        processor_func,
        remove_columns=full_train_data.column_names,
        num_proc=args.num_proc
    )
    full_valid_data = full_valid_data.map(
        processor_func,
        remove_columns=full_valid_data.column_names,
        num_proc=args.num_proc
    )
    print(f"Removing audio files longer than {args.max_length} secs...")
    full_train_data = remove_long_data(full_train_data, args.max_length)
    full_valid_data = remove_long_data(full_valid_data, args.max_length)
    print("Dataset lengths to be trained and tested:")
    print("Train:", len(full_train_data))
    print("Valid:", len(full_valid_data))
    print("Preprocessing done")

    print("Creating the data collator")
    data_collator = DataCollatorCTCWithPadding(processor=processor_ipa, padding=True)
    print("Data collator created")
    
    # Model
    print("Defining the model...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor_ipa.tokenizer.pad_token_id,
        vocab_size=len(processor_ipa.tokenizer)
        )
    print("Model defined")

    # Freeze the feature extractor so that it won't be changed by the fine-tuning
    print("Freezing the feature extractor...") 
    model.freeze_feature_encoder()
    print("Feature extractor frozen")

    model_dir = output_dir / "wav2vec2-large-xlsr-{}-ipa".format("".join(args.corpus))

    print("Running garbage collection before training")
    gc.collect()
    torch.cuda.empty_cache()

    # Training
    print("Beginning the training...") 
    training_args = TrainingArguments(
        output_dir=model_dir,
        group_by_length=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=full_train_data,
        eval_dataset=full_valid_data,
        tokenizer=processor_ipa.feature_extractor,
        )
    
    trainer.train()
    trainer.evaluate()
    trainer.save_state()
    trainer.save_model()
    # You also need to save the tokenizer in order to save the model
    tokenizer_ipa.save_pretrained(model_dir)
    # trainer.push_to_hub(repo_name="wav2vec2-ipa")

if __name__ == "__main__": 
    main_cli()