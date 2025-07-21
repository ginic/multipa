from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib.resources
import logging
import os
from pathlib import Path
from typing import Literal

import datasets

logger = logging.getLogger(__name__)

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


def extract_all_chars_ipa(batch: dict) -> dict:
    """Returns the all characters which appear in the "ipa" field in the
    batch. Used to build vocabulary if there is no tokenization or whitespace
    delimiting around phonetic symbols.
    """
    all_text = "".join(batch["ipa"])
    return {"vocab": list(set(all_text))}


def extract_whitespace_delimited_symbols(batch: dict) -> dict:
    """Returns the whitespace delimited strings that appear in the "ipa" field
    in the batch. Used to build vocabulary when there is tokenization present.
    """
    all_text = " ".join(batch["ipa"])
    symbols = set(all_text.split())
    return {"vocab": list(symbols)}


def remove_space(batch: dict, col_key: str) -> dict:
    """Returns the batch with whitespace removed in the
    value at batch[col_key].
    """
    ipa = batch[col_key]
    ipa = ipa.split()
    ipa = "".join(ipa)
    batch[col_key] = ipa
    return batch


def replace_none(batch: dict, col_key) -> dict:
    """Replaces any None value for the given col_key with the empty string '' instead."""
    ipa = batch[col_key]
    if ipa is None:
        batch[col_key] = EMPTY_TRANSCRIPTION
    return batch


def clean_text(batch: dict, text_key="ipa", is_remove_space=True):
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


def validate_dataset_files_match(raw_data, ipa_data, key: str, is_check_basename: bool = False) -> None:
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


def concatenate_common_voice(datasetlist: list[datasets.Dataset]) -> datasets.Dataset:
    """Concatenate more than one datasets from Common Voice."""
    init_data = datasetlist[0]
    for d in datasetlist:
        assert d.features.type == init_data.features.type
    concatenated = datasets.concatenate_datasets(datasetlist)
    return concatenated


def join_column(
    left_dataset,
    right_dataset,
    on_key: str,
    right_col: str,
    is_check_basename: bool = False,
    additional_check_col: str | None = "sentence",
) -> datasets.Dataset:
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


def load_common_voice_split(
    language: str,
    quality_filter: bool,
    split: str,
    huggingface_split: str,
    data_dir: str | os.PathLike,
    json_filename: str,
    cache_dir: str | os.PathLike,
    num_proc: int,
    dataset_name: str = "mozilla-foundation/common_voice_11_0",
) -> datasets.Dataset:
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
    ipa_dataset = datasets.load_dataset("json", data_files=str(Path(data_dir) / json_filename), split=split)
    raw_audio = datasets.load_dataset(
        dataset_name, language, split=huggingface_split, num_proc=num_proc, cache_dir=cache_dir
    )

    full_dataset = join_column(raw_audio, ipa_dataset, "path", "ipa", is_check_basename=True)

    # Remove Tamil sentences containing "ச"
    if language == "ta":
        full_dataset = full_dataset.filter(lambda batch: "ச" not in batch["sentence"], num_proc=num_proc)

    if quality_filter:
        full_dataset = full_dataset.filter(lambda batch: batch["down_votes"] == 0, self.num_proc)

    return full_dataset


def load_librispeech_split(
    split: str,
    huggingface_split: str,
    data_dir: str | os.PathLike,
    json_filename: str,
    cache_dir: str | os.PathLike,
    num_proc: int,
    dataset_name: str = "librispeech_asr",
) -> datasets.Dataset:
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
    ipa_dataset = datasets.load_dataset("json", data_files=str(Path(data_dir) / json_filename), split=split)
    ipa_dataset = ipa_dataset.rename_column("text", "sentence")

    raw_audio = datasets.load_dataset(dataset_name, split=huggingface_split, num_proc=num_proc, cache_dir=cache_dir)
    raw_audio = raw_audio.rename_column("text", "sentence")

    # Join in IPA data by matching file name
    full_dataset = join_column(raw_audio, ipa_dataset, "file", "ipa")
    full_dataset = full_dataset.rename_column("file", "path")
    return full_dataset


def load_buckeye_split(corpus_root_dir: str | os.PathLike, split: str) -> datasets.Dataset:
    dataset_split = datasets.load_dataset("audiofolder", data_dir=corpus_root_dir, split=split)
    # Output data has duplicates despite following the format from HuggingFace documentation, so deduplicate based on utterance id
    deduplicated_df = dataset_split.to_pandas().drop_duplicates("utterance_id")
    return datasets.Dataset.from_pandas(deduplicated_df)


@dataclass
class SimpleSampler:
    seed: int
    num_samples: int


@dataclass
class SubsetSampler:
    seed: int
    num_samples: list[int]
    subset_identifiers: list[str]

    def __post_init__(self):
        if len(self.num_samples) != len(self.subset_identifiers):
            raise ValueError(f"Sampling argument must match length of {self.subset_identifiers}")


class CorpusPreprocessor(ABC):
    """Abstract class with defined behavior for preparing data from a specific
    source corpus for training
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str | os.PathLike,
        cache_dir: str | os.PathLike,
        train_sampler: SimpleSampler | SubsetSampler,
        val_sampler: SimpleSampler | SubsetSampler,
        num_proc: int,
        file_suffix: str,
        unused_columns: None | list[str] = None,
        is_remove_spaces: bool = False,
        vocab_resource_file: None | str = None,
    ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.num_proc = num_proc
        self.suffix = file_suffix
        self.unused_columns = unused_columns
        self.is_remove_spaces = is_remove_spaces
        self.vocab_resource_file = vocab_resource_file

    @abstractmethod
    def get_train_split(self) -> datasets.Dataset:
        pass

    @abstractmethod
    def get_validation_split(self) -> datasets.Dataset:
        pass

    def clean_ipa_transcription(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """Replace missing IPA transcriptions with the empty string and remove spaces
        if the corpus requires it.

        This is closely related to vocabulary creation, as symbols may be whitespace delimited
        or not, depending on how the IPA transcriptions were originally obtained.
        """
        return dataset.map(lambda x: clean_text(x, is_remove_space=self.is_remove_spaces), num_proc=self.num_proc)

    def create_vocabulary(self, *datasets: datasets.Dataset) -> dict[str, int]:
        """Build the vocabulary from the dataset and any specified vocab resource files.

        Returns:
            dict[str, int]: symbol to index dictionary expected by HuggingFace
        """
        final_vocab = set()
        if self.is_remove_spaces:
            token_func = extract_all_chars_ipa
        else:
            token_func = extract_whitespace_delimited_symbols
        for d in datasets:
            d_vocab = d.map(
                token_func,
                batched=True,
                keep_in_memory=True,
                remove_columns=d.column_names,
                num_proc=self.num_proc,
            )
            final_vocab = final_vocab | set(d_vocab["vocab"])

        if self.vocab_resource_file is not None:
            vocab_file = importlib.resources.files("multipa.resources").joinpath(self.vocab_resource_file)
            vocab_from_file = set([line.strip() for line in vocab_file.read_text().splitlines()])

            # Check and warn about symbol mismatches
            symbols_missing_in_dataset = vocab_from_file - final_vocab
            if len(symbols_missing_in_dataset) > 0:
                logger.warning(
                    "%s symbol(s) present in vocabulary didn't appear in data: %s",
                    len(symbols_missing_in_dataset),
                    symbols_missing_in_dataset,
                )

            symbols_not_in_file = final_vocab - vocab_from_file
            if len(symbols_not_in_file) > 0:
                logger.warning(
                    "%s symbol(s) in dataset not in vocab file: %s", len(symbols_not_in_file), symbols_not_in_file
                )

            final_vocab = final_vocab | vocab_from_file

        vocab_dict_ipa = {v: k for k, v in enumerate(final_vocab)}
        vocab_dict_ipa[UNKNOWN_TOKEN] = len(vocab_dict_ipa)
        vocab_dict_ipa[PADDING_TOKEN] = len(vocab_dict_ipa)
        return vocab_dict_ipa

    def remove_unused_columns(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """Returns the dataset dropping any columns not used by the CorpusPreprocessor"""
        if self.unused_columns is not None:
            cols_to_remove = list(set(self.unused_columns).intersection(dataset.column_names))
            logger.debug("Removing unnecessary columns: %s", cols_to_remove)
            return dataset.remove_columns(cols_to_remove)
        return dataset


class BuckeyePreprocessor(CorpusPreprocessor):
    DATASET_NAME = "buckeye"
    COLS_TO_DROP = [
        "speaker_gender",
        "speaker_age_range",
        "interviewer_gender",
        "buckeye_transcript",
        "duration",
        "utterance_id",
        "text",
    ]

    def __init__(
        self,
        data_dir: str | os.PathLike,
        cache_dir: str | os.PathLike,
        train_sampler: SimpleSampler,
        val_sampler: SimpleSampler,
        num_proc: int,
        file_suffix: str,
        min_length: float = 0.1,
        max_length: float = 12,
        speaker_restriction: None | list[str] = None,
        percent_female: float = 0.5,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.percent_female = percent_female
        if speaker_restriction is None:
            self.speaker_restriction = []
        else:
            self.speaker_restriction = speaker_restriction
        super().__init__(
            BuckeyePreprocessor.DATASET_NAME,
            data_dir,
            cache_dir,
            train_sampler,
            val_sampler,
            num_proc,
            file_suffix,
            unused_columns=self.COLS_TO_DROP,
            is_remove_spaces=False,
        )

    def _sample_gender_subset(self, dataset, num_samples: int, seed: int, gender_value: Literal["m", "f"]):
        gender_examples = dataset.filter(lambda x: x["speaker_gender"] == gender_value, num_proc=self.num_proc)
        gender_examples = gender_examples.shuffle(seed=seed).select(range(min(num_samples, len(gender_examples))))
        logger.info("Number of examples from 'speaker_gender'='%s': %s", gender_value, len(gender_examples))
        logger.info(
            "Total duration(seconds) of examples from 'speaker_gender'='%s': %s",
            gender_value,
            sum(gender_examples["duration"]),
        )

        return gender_examples

    def get_train_split(self):
        train_data = load_buckeye_split(self.data_dir, "train")
        # For Buckeye, it's important to remove examples that are too long/short first
        # to maintain the specified gender split and restrictions to certain speakers

        logger.info(
            "Filtering Buckeye training data with sample duration >= %s, < %s", self.min_length, self.max_length
        )
        train_data = train_data.filter(lambda x: x["duration"] < self.max_length, num_proc=self.num_proc)
        train_data = train_data.filter(lambda x: x["duration"] >= self.min_length, num_proc=self.num_proc)

        # Handle restrictions to particular individuals
        if self.speaker_restriction:
            logger.info("Filtering Buckeye training to speaker ids %s", self.speaker_restriction)
            train_data = train_data.filter(
                lambda x: x["speaker_id"] in self.speaker_restriction, num_proc=self.num_proc
            )

        logger.info("Buckeye train dataset size after filtering by duration and speaker id: %s", len(train_data))

        # Select equal numbers of examples matching the gender split
        logger.info(
            "Sampling Buckeye training data by gender split with %s ratio female speakers", self.percent_female
        )
        num_female_examples = int(self.train_sampler.num_samples * self.percent_female)
        female_examples = self._sample_gender_subset(train_data, num_female_examples, self.train_sampler.seed, "f")

        num_male_examples = self.train_sampler.num_samples - num_female_examples
        male_examples = self._sample_gender_subset(train_data, num_male_examples, self.train_sampler.seed, "m")

        full_train_data = datasets.concatenate_datasets([female_examples, male_examples])
        logger.info("Full train dataset size: %s", len(full_train_data))

        return self.remove_unused_columns(full_train_data)

    def get_validation_split(self) -> datasets.Dataset:
        valid_data = load_buckeye_split(self.data_dir, "validation")
        valid_limit = min(self.val_sampler.num_samples, len(valid_data))
        # Shuffle because datasets are often ordered by speaker and you want a variety of speakers.
        full_valid_data = valid_data.shuffle(seed=self.val_sampler.seed).select(range(valid_limit))
        return self.remove_unused_columns(full_valid_data)


class CommonVoicePreprocessor(CorpusPreprocessor):
    COLS_TO_DROP = [
        "accent",
        "age",
        "client_id",
        "down_votes",
        "gender",
        "locale",
        "segment",
        "up_votes",
    ]
    VOCAB_RESOURCE = "full_vocab_ipa.txt"

    def __init__(
        self,
        data_dir: str | os.PathLike,
        cache_dir: str | os.PathLike,
        train_sampler: SubsetSampler,
        val_sampler: SubsetSampler,
        num_proc: int,
        file_suffix: str,
        dataset_name: str = "mozilla-foundation/common_voice_11_0",
        quality_filter: bool = True,
    ):
        self.quality_filter = quality_filter
        super().__init__(
            dataset_name,
            data_dir,
            cache_dir,
            train_sampler,
            val_sampler,
            num_proc,
            file_suffix,
            unused_columns=self.COLS_TO_DROP,
            vocab_resource_file=self.VOCAB_RESOURCE,
        )

    def _get_split(self, split_name: str, huggingface_split: str, sampler: SubsetSampler):
        data_list = []
        for i, lang in enumerate(sampler.subset_identifiers):
            num_samples = sampler.num_samples[i]
            logger.info(
                "Retrieving max %s samples for language '%s' from %s",
                num_samples,
                lang,
                self.dataset_name,
            )
            data = load_common_voice_split(
                lang,
                self.quality_filter,
                split_name,
                huggingface_split,
                self.data_dir,
                f"{lang}_train{self.suffix}.json",
                self.cache_dir,
                self.num_proc,
                self.dataset_name,
            )
            sample_limit = min(num_samples, len(data))
            logger.info("Sampling %s examples for language '%s'", sample_limit, lang)
            data = data.shuffle(seed=sampler.seed).select(range(sample_limit))
            data_list.append(data)

        logger.debug("Concatentating CommonVoice voice language subsets to final dataset")
        full_data = concatenate_common_voice(data_list)
        return self.remove_unused_columns(full_data)

    def get_train_split(self):
        return self._get_split("train", "train", self.train_sampler)

    def get_validation_split(self):
        return self._get_split("valid", "validation", self.val_sampler)


class LibriSpeechPreprocessor(CorpusPreprocessor):
    DATASET_NAME = "librispeech_asr"
    COLS_TO_DROP = [
        "speaker_id",
        "chapter_id",
        "id",
    ]
    VOCAB_RESOURCE = "full_vocab_ipa.txt"

    def __init__(
        self,
        data_dir: str | os.PathLike,
        cache_dir: str | os.PathLike,
        train_sampler: SimpleSampler,
        val_sampler: SimpleSampler,
        num_proc: int,
        file_suffix: str,
    ):
        super().__init__(
            LibriSpeechPreprocessor.DATASET_NAME,
            data_dir,
            cache_dir,
            train_sampler,
            val_sampler,
            num_proc,
            file_suffix,
            unused_columns=self.COLS_TO_DROP,
            vocab_resource_file=self.VOCAB_RESOURCE,
        )

    def _get_split(self, split_name, huggingface_split, sampler, json_filename):
        dataset = load_librispeech_split(
            split_name,
            huggingface_split,
            self.data_dir,
            json_filename,
            self.cache_dir,
            self.num_proc,
            self.DATASET_NAME,
        )
        limit = min(sampler.num_samples, len(dataset))
        dataset = dataset.shuffle(seed=sampler.seed).select(range(limit))
        return self.remove_unused_columns(dataset)

    def get_train_split(self):
        return self._get_split("train", "train.clean.100", self.train_sampler, f"en_train{self.suffix}.json")

    def get_validation_split(self):
        return self._get_split("valid", "validation.clean", self.val_sampler, f"en_valid{self.suffix}.json")
