import argparse
import gc
import importlib.resources
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

from dataclasses import dataclass
from datasets import Audio, concatenate_datasets
import evaluate
import numpy as np
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import torch

from multipa.data_utils import UNKNOWN_TOKEN, PADDING_TOKEN, BUCKEYE_KEY, COMMONVOICE_KEY, LIBRISPEECH_KEY, clean_text, load_buckeye_split, load_common_voice_split, load_librispeech_split

# Transformers gives a lot of warnings and it's hard to check training progress, so only print them once
warnings.filterwarnings("ignore", category=UserWarning)

PHONE_ERRORS_COMPUTER = evaluate.load("ginic/phone_errors")

# Ignore Forvo for now, I don't have access to the data
# from multipa.add_forvo import add_language


# Following class comes from https://github.com/huggingface/transformers/blob/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/examples/research_projects/wav2vec2/run_asr.py#L81
# and https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb#scrollTo=Slk403unUS91
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
    all_text = "".join(batch["ipa"])
    return {"vocab": list(set(all_text))}


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
    dftest = dftest.drop(columns = ['len'],)
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


def create_vocabulary(*datasets, use_resource_vocab=True):
    """Determines the vocabulary of IPA characters needed for the model.

    Returns:
        dict: vocab -> index
    """
    vocab_set = set()
    for d in datasets:
        d_vocab = d.map(
            extract_all_chars_ipa,
            batched=True,
            keep_in_memory=True,
            remove_columns=d.column_names
        )
        vocab_set = vocab_set | set(d_vocab["vocab"])

    # Add in data from resources file
    if use_resource_vocab: 
        all_vocab_file = importlib.resources.files("multipa.resources").joinpath("full_vocab_ipa.txt")
        new_vocab = set([l.strip() for l in all_vocab_file.read_text().splitlines()])
        vocab_set = vocab_set | new_vocab
        
    vocab_dict_ipa = {v: k for k, v in enumerate(vocab_set)}
    vocab_dict_ipa[UNKNOWN_TOKEN] = len(vocab_dict_ipa)
    vocab_dict_ipa[PADDING_TOKEN] = len(vocab_dict_ipa)
    return vocab_dict_ipa


def compute_metrics(pred, processor):
    """Returns metrics results for at incremental times in training
    """    
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    return PHONE_ERRORS_COMPUTER.compute(predictions=pred_str, references=label_str)


def check_gpus(expected_gpus: int = 0):
    """Validates that the epxected number of GPUs is available and prints memory usage.
    Set up for tracking GPU usage if using CUDA, see https://huggingface.co/docs/transformers/v4.18.0/en/performance

    Args:
        expected_gpus (int): A number (>0) of gpus to check for. If this exact number isn't found, raises a RuntimeError. 
    """
    import pynvml
    pynvml.nvmlInit()
    num_gpus = torch.cuda.device_count()

    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU {i} handle: {handle}")
        print(f"GPU {i} memory occupied: {info.used//1024**2} MB.")

    if expected_gpus > 0 and expected_gpus != num_gpus:
        raise RuntimeError(f"Expected {expected_gpus} gpus, but found only {num_gpus}.")


def is_valid_sample(batch):
    audio = batch["audio"]
    if len(audio["array"]) < 1600:  # 0.1 seconds at 16kHz
        return False
    if len(batch["ipa"]) == 0:
        return False
    return True


def is_valid_post_tokenization(batch):
    return (
        "input_values" in batch and len(batch["input_values"]) > 0 and
        "labels" in batch and len(batch["labels"]) > 0
    )

def main_cli():
    parser = argparse.ArgumentParser(description="Trains the speech recognition model. Specify corpus, "\
                                     "model training parameters and language details if needed. ")
    
    # Model parameters
    parser.add_argument("-e", "--num_train_epochs", type=int, default=10,
                        help="Specify the number of train epochs. By default it's set to 10.")
    parser.add_argument("-lr", "--learning_rate", type=float, default= 3e-4, 
                        help="The learning rate for the optimizer during training")
    parser.add_argument("-bs", "--per_device_train_batch_size", type=int, default=2, 
                        help="The batch size per GPU/CPU for training, defaults to 2.")
    parser.add_argument("-ga", "--gradient_accumulation_steps", type=int, default=4, 
                        help="The number of gradient accumulation steps during training, defaults to 4")
    parser.add_argument("-ml", "--max-length", type=float, default=12, help="Maximum audio length of training & validation samples in seconds. Defaults to 12.")
    parser.add_argument("-mt", "--mask_time_length", type=int, default=10, 
                        help="Mask time length for the model. If you know your training data contains audio less than 0.2 seconds, make the mask time length small. Defaults to 10.")
    parser.add_argument("-g", "--use_gpu", action="store_true", help="Use this flag if a GPU is available for training.")
    parser.add_argument("-bm", "--base_model", type=str, default="facebook/wav2vec2-large-xlsr-53", choices=["facebook/wav2vec2-xls-r-300m", "facebook/wav2vec2-large-xlsr-53"],
                        help="The base pre-trained speech recognition model to use.")

    # Data processing parameters used for all corpora
    parser.add_argument("--num_proc", type=int, default=8,
                        help="Specify the number of CPUs for preprocessing. Default set to 8.")
    parser.add_argument("--num_gpus", type=int, default=0, 
                        help="If you're using GPUs and would like to check that they are all found correctly, set this to the expected number of GPUs >=1. Otherwise this parameter is ignored.")
    parser.add_argument("-ns", "--no_space", action='store_true',
                        help="Use this flag remove spaces in IPA transcription.") 
    parser.add_argument("-o", "--output_dir", type=str, 
                        help="Specify the directory to save files for vocab, stats and trained models.")
    parser.add_argument("-s", "--suffix", type=str, default="",
                        help="Optional suffix to use when naming vocab and model folders")
    parser.add_argument("-ts", "--train_seed", type=int, default=7, 
                        help="Random seed for selecting a subset of training data. Defaults to 7.")
    parser.add_argument("-vs", "--val_seed", type=int, default=99, 
                        help="Random seed for selecting a subset of validation data. Defaults to 99.")
    # This is a bit confusing, but it's basically reading the train/test splits from the preprocessing output. 
    parser.add_argument("-dd", "--data_dir", type=str, default="data_new",
                        help="Specify the directory path for the training/validation data files." \
                        "Default is set to `data_new`.")

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
    comm_voice_subparser.add_argument("-tr", "--train_samples", nargs="+", type=int,
                        help="Specify the number of samples to be used as the training data for each language. " \
                        "For example, if you want to use 1000, 2000, 3000 training samples for Japanese, Polish, " \
                        "and Maltese, then specify as -l ja pl mt -tr 1000 2000 3000." \
                        "You can type an irrationally large number to pick up the maximum value.")
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
    buckeye_subparser.add_argument("-pf", "--percent_female", type=float, default=0.5, 
                                   help="The percentage (as float) of training examples that should come from female speakers. Defaults to 0.5")
    buckeye_subparser.add_argument("-sr", "--speaker_restriction", nargs="*", type=str, 
                                   help="A list of speakers ids to restrict the training data to.")
    buckeye_subparser.add_argument("--min-length", type=float, default=0.1, help="Minimum sample length for training samples data in seconds. Only used for buckeye. Defaults to 0.1.")
    
        
    args = parser.parse_args()

    # Check that the desired number of gpus is available
    if args.use_gpu:
        check_gpus(args.num_gpus)
           
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
        
    # Set up corpus stats tracking file
    stats_file = output_dir / f"stats_train_valid{args.suffix}.txt"
    with open(stats_file, "w") as f:
        f.write("corpus lang train valid\n")

    with open(output_dir / f"training_args.json", "w") as train_args_json:
        json.dump(vars(args), train_args_json)

    final_results_to_write = {}

    if args.corpus == LIBRISPEECH_KEY:
        dataset_name = "librispeech_asr"
        train_data = load_librispeech_split("train", "train.clean.100", args.data_dir, f"en_train{args.suffix}.json", 
                                            args.cache_dir, args.num_proc, dataset_name)
        valid_data = load_librispeech_split("valid", "validation.clean", args.data_dir, f"en_valid{args.suffix}.json", 
                                            args.cache_dir, args.num_proc, dataset_name)
        # Shuffle and clip to the specified sample size using datasets's Dataset.select(). 
        # Shuffle because datasets are often ordered by speaker and you want a variety of speakers.
        train_limit = min(args.train_samples, len(train_data))
        valid_limit = min(args.val_samples, len(valid_data))
       
        full_train_data = train_data.shuffle(seed=args.train_seed).select(range(train_limit))
        full_valid_data = valid_data.shuffle(seed=args.val_seed).select(range(valid_limit))

        with open(stats_file, "a") as f:
            f.write(f"{dataset_name} en {len(full_train_data)} {len(full_valid_data)}\n")

    elif args.corpus == BUCKEYE_KEY: 
        dataset_name = "buckeye"
        train_data = load_buckeye_split(args.data_dir, "train")

        # For Buckeye, it's important to remove examples that are too long/short first 
        # to maintain the specified gender split and restrictions to certain speakers
        train_data = train_data.filter(lambda x: x["duration"] < args.max_length)
        train_data = train_data.filter(lambda x: x["duration"] >= args.min_length)

        # Handle restrictions to particular individuals
        if args.speaker_restriction:
            train_data = train_data.filter(lambda x: x["speaker_id"] in args.speaker_restriction)
        
        print("Sampling from Buckeye train dataset with length after filtering:", len(train_data))
        
        # Select equal numbers of examples matching the gender split
        num_female_examples = int(args.train_samples * args.percent_female)
        female_examples = train_data.filter(lambda x: x["speaker_gender"]=="f")
        female_examples = female_examples.shuffle(seed=args.train_seed).select(range(min(num_female_examples, len(female_examples))))
        final_results_to_write["train_num_female_examples"] = len(female_examples)
        final_results_to_write["train_duration_female_examples"] = sum(female_examples["duration"])
        print("Number of examples from female speakers:", final_results_to_write["train_num_female_examples"])
        print("Total duration (seconds) of examples from female speakers :", final_results_to_write["train_duration_female_examples"])

        num_male_examples = args.train_samples - num_female_examples
        male_examples = train_data.filter(lambda x: x["speaker_gender"]=="m")
        male_examples = male_examples.shuffle(seed=args.train_seed).select(range(min(num_male_examples, len(male_examples))))
        final_results_to_write["train_num_male_examples"] = len(male_examples)
        final_results_to_write["train_duration_male_examples"] = sum(male_examples["duration"])
        print("Number of examples from male speakers:", final_results_to_write["train_num_male_examples"])
        print("Total duration (seconds) of examples from male speakers :", final_results_to_write["train_duration_male_examples"])

        full_train_data = concatenate_datasets([female_examples, male_examples])
        print("Full train dataset size:", len(full_train_data))
        
        valid_data = load_buckeye_split(args.data_dir, "validation")
        valid_limit = min(args.val_samples, len(valid_data))
        # Shuffle because datasets are often ordered by speaker and you want a variety of speakers.
        full_valid_data = valid_data.shuffle(seed=args.val_seed).select(range(valid_limit))

        with open(stats_file, "a") as f:
            f.write(f"{dataset_name} en {len(full_train_data)} {len(full_valid_data)}\n")
    
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
                                                 args.data_dir, f"{lang}_train{args.suffix}.json", args.cache_dir, args.num_proc, dataset_name)
            valid_data = load_common_voice_split(lang, args.quality_filter, "valid", "validation", 
                                                 args.data_dir, f"{lang}_valid{args.suffix}.json", args.cache_dir, args.num_proc, dataset_name)

            # Shuffle and clip to the specified sample size using datasets's Dataset.select(). 
            # Shuffle because datasets are often ordered by speaker and you want a variety of speakers.
            train_limit = min(train_sample, len(train_data))
            valid_limit = min(valid_sample, len(valid_data))
            train_data = train_data.shuffle(seed=args.train_seed).select(range(train_limit))
            valid_data = valid_data.shuffle(seed=args.val_seed).select(range(valid_limit))
            
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

    # Remove unnecessary columns - have to do this using remove_columns because select_columns wasn't available in older HF versions
    unnecessary_columns = [
        "accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes", # for Common Voice
        "speaker_id", "chapter_id", "id", #for librispeech
        "speaker_gender", "speaker_age_range", "interviewer_gender", "buckeye_transcript", "duration", "utterance_id", "text", # for Buckeye
        ]
    columns_to_remove = set(unnecessary_columns).intersection(full_train_data.column_names)
    print("Removing unnecessary columns:", columns_to_remove)
    full_train_data = full_train_data.remove_columns(columns_to_remove)
    full_valid_data = full_valid_data.remove_columns(columns_to_remove)
    print("Unnecessary columns removed. Data preview:")
    print(full_train_data[0])
    assert full_train_data.features.type == full_valid_data.features.type
    
    full_train_data = full_train_data.map(lambda x: clean_text(x, is_remove_space = args.no_space))
    full_valid_data = full_valid_data.map(lambda x: clean_text(x, is_remove_space = args.no_space))



    # Preprocessing 
    print("Creating vocabulary...")
    vocab_dict_ipa = create_vocabulary(full_train_data, full_valid_data)

    print("Writing vocab json files...")
    # Don't forget to change the file name when you use different languages,
    # otherwise the vocab file will be lost
    vocab_file = output_dir / f"{args.corpus}_ipa_vocab{args.suffix}.json"
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
    # See notes on feature extractor settings at 
    # https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb#scrollTo=mYcIiR2FQ96i
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                 sampling_rate=16_000,
                                                 padding_value=0.0,
                                                 do_normalize=True, # zero-mean-unit variance normalized
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

    # Final filter to check that the samples are valid
    print("Filtering the dataset to remove invalid samples...")
    full_train_data = full_train_data.filter(is_valid_sample)
    full_valid_data = full_valid_data.filter(is_valid_sample)

    print("Preprocessing the dataset...")
    # Critically, this assigns the "labels" values 
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

    # Check that the post-tokenization data is valid
    print("Checking that the post-tokenization data is valid...")
    full_train_data = full_train_data.filter(is_valid_post_tokenization)
    full_valid_data = full_valid_data.filter(is_valid_post_tokenization)
    print("Validation done. Data preview:")
    print(full_train_data[0])

    print(f"Removing audio files longer than {args.max_length} secs...")
    full_train_data = remove_long_data(full_train_data, args.max_length)
    full_valid_data = remove_long_data(full_valid_data, args.max_length)

    # Shuffle the dataset
    print("Shuffling the dataset...")
    full_train_data = full_train_data.shuffle(seed=42).flatten_indices(keep_in_memory=True)
    full_valid_data = full_valid_data.shuffle(seed=35).flatten_indices(keep_in_memory=True)
    print("Shuffling done")


    print("Dataset lengths to be trained and tested:")
    print("Train:", len(full_train_data))
    print("Valid:", len(full_valid_data))
    print("Preprocessing done")

    print("Creating the data collator")
    data_collator = DataCollatorCTCWithPadding(processor=processor_ipa, padding=True)
    print("Data collator created")
    
    # Model
    print(f"Defining the model from pretrained '{args.base_model}'")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        # TODO Look into masking in this model. How does it work, what's the trade-off in tweaking the probability and length
        # TODO This needs to be updated to a later transformers version, see https://github.com/huggingface/transformers/issues/15366
        mask_time_prob=0.05,
        mask_time_length=args.mask_time_length, # If this is too big, you'll get `ValueError: `mask_length` has to be smaller than `sequence_length`...`, so if you have very short audio in training data, reduce the value
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor_ipa.tokenizer.pad_token_id,
        vocab_size=len(processor_ipa.tokenizer)
        )
    # This should fix issues with validation loss going to inf, https://discuss.huggingface.co/t/wav2vec2-how-to-correct-for-nan-in-training-and-validation-loss/6089/2
    model.config.ctc_zero_infinity = True
    print("Model defined")

    # Freeze the feature extractor so that it won't be changed by the fine-tuning
    print("Freezing the feature extractor...")
    model.freeze_feature_encoder() 
    print("Feature extractor frozen")

    base_model_id = args.base_model.split("/")[1]
    model_dir = output_dir / f"{base_model_id}-{args.corpus}-ipa{args.suffix}"

    print("Running garbage collection before training")
    gc.collect()
    torch.cuda.empty_cache()

    # How will metrics be computed? 
    def eval_compute_metrics(x):
        eval_results = compute_metrics(x, processor_ipa)
        # Don't keep example level results
        del eval_results["phone_error_rates"]
        del eval_results["feature_error_rates"]
        del eval_results["phone_feature_error_rates"]
        return eval_results


    # Training
    # See https://huggingface.co/docs/transformers/v4.18.0/en/performance#optimizer for options on tuning and performance
    print("Beginning the training...") 
    training_args = TrainingArguments(
        output_dir=model_dir,
        group_by_length=True,
        # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        #gradient_checkpointing=True, # Can use this if memory is a problem, but training will be slower
        #optim="adafactor", #Can use if memory is a problem, but convergence might be slower
        num_train_epochs=args.num_train_epochs,
        fp16=False, # False to keep memory usage down 
        # Defaults are fine for logging and evaluation, but if you'd like to save time, you can 
        evaluation_strategy="epoch",
        save_strategy="epoch",
        #save_steps=500,
        #eval_steps=500,
        logging_steps=100,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=full_train_data,
        eval_dataset=full_valid_data,
        compute_metrics=eval_compute_metrics,
        tokenizer=processor_ipa.feature_extractor,
        )
    
    train_result = trainer.train()
    print("Training finished:", train_result)

    if args.use_gpu:
        try:
            check_gpus(args.num_gpus)
        except RuntimeError as e:
            print("WARNING:", e)
    
    eval_results = trainer.evaluate()
    final_results_to_write.update(eval_results)
    print("Final evaluation results:", eval_results)
    with open(output_dir / "final_evaluation.json", 'w') as eval_json:
        json.dump(final_results_to_write, eval_json)

    
    trainer.save_state()
    trainer.save_model()
    # You also need to save the tokenizer in order to save the model
    tokenizer_ipa.save_pretrained(model_dir)
    # trainer.push_to_hub(repo_name="wav2vec2-ipa")

if __name__ == "__main__": 
    main_cli()
