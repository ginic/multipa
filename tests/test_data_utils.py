import datasets

from multipa.data_utils import (
    clean_text,
    BuckeyePreprocessor,
    CommonVoicePreprocessor,
    LibriSpeechPreprocessor,
    SimpleSampler,
)


import pytest

from datasets import Dataset


@pytest.fixture(scope="session")
def buckeye_preprocessor():
    return BuckeyePreprocessor(
        data_dir=None,
        cache_dir=None,
        train_sampler=SimpleSampler(10, 6),
        val_sampler=SimpleSampler(99, 4),
        num_proc=1,
        file_suffix="buckeye",
    )


@pytest.fixture(scope="session")
def mock_librispeech():
    mock_dict = {
        "audio": [{}] * 5,
        "chapter_id": [1234] * 5,
        "id": ["9876"] * 5,
        "speaker_id": [9876] * 5,
        "text": ["THIS IS A TEST"] * 5,
        "ipa": ["ðɪsɪzətɛst"] * 5,
    }
    return Dataset.from_dict(mock_dict)


@pytest.fixture(scope="session")
def mock_common_voice():
    mock_dict = {
        "audio": [{}] * 5,
        "locale": ["en"] * 5,
        "sentence": ["This is a test"] * 5,
        "ipa": ["ðɪsɪzətɛst"] * 5,
        "age": ["teens", "twenties", "thirties", "forties", "fifties"],
        "gender": ["male", "female", "male", "female", "male"],
        "down_votes": [0, 0, 0, 4, 5],
        "up_votes": [1, 3, 2, 0, 1],
    }
    return Dataset.from_dict(mock_dict)


@pytest.fixture(scope="session")
def mock_buckeye():
    mock_dict = {
        "utterance_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "duration": [0.09, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 0.09],
        "buckey_transcript": ["DH IH S IH Z AH T EH1 S T"] * 12,
        "text": ["this is a test"] * 12,
        "ipa": ["ð ɪ s ɪ z ə t ɛ s t"] * 12,
        "speaker_id": [f"S{i:02d}" for i in range(1, 13)],
        "speaker_gender": ["m"] * 6 + ["f"] * 6,
        "speaker_age_range": ["y", "o"] * 6,
    }
    return Dataset.from_dict(mock_dict)


def test_clean_text_dataset():
    examples = [
        {"text": "hello", "ipa": "hɛlo"},
        {"text": "U U", "ipa": None},
        {"text": "this is a test", "ipa": "ðɪs ɪz ə tɛst"},
    ]
    input_dataset = datasets.Dataset.from_list(examples)
    output_dataset = input_dataset.map(lambda x: clean_text(x, is_remove_space=True))
    assert len(output_dataset) == 3
    assert {"text": "hello", "ipa": "hɛlo"} in output_dataset
    assert {"text": "U U", "ipa": ""} in output_dataset
    assert {"text": "this is a test", "ipa": "ðɪsɪzətɛst"} in output_dataset


def test_clean_text_keep_spaces():
    examples = [
        {"text": "hello", "ipa": "h ɛ l o"},
        {"text": "U U", "ipa": None},
        {"text": "this is a test", "ipa": "ðɪs ɪz ə tɛst"},
    ]
    input_dataset = datasets.Dataset.from_list(examples)
    output_dataset = input_dataset.map(lambda x: clean_text(x, is_remove_space=False))
    assert len(output_dataset) == 3
    assert {"text": "hello", "ipa": "h ɛ l o"} in output_dataset
    assert {"text": "U U", "ipa": ""} in output_dataset
    assert {"text": "this is a test", "ipa": "ðɪs ɪz ə tɛst"} in output_dataset


def test_buckeye_preprocessor_init(buckeye_preprocessor):
    assert buckeye_preprocessor.dataset_name == "buckeye"
    assert buckeye_preprocessor.unused_columns == BuckeyePreprocessor.COLS_TO_DROP
    assert buckeye_preprocessor.vocab_resource_file == "buckeye_ipa_inventory.txt"


@pytest.mark.parametrize(
    "percent_f, expected_f, expected_m",
    [(0.5, 5, 5), (0.7, 5, 3), (0.2, 2, 5)],
)
def test_buckeye_sample_gender(percent_f, expected_f, expected_m, mock_buckeye):
    buckeye_preprocessor = BuckeyePreprocessor(
        data_dir=None,
        cache_dir=None,
        train_sampler=SimpleSampler(10, 10),
        val_sampler=SimpleSampler(99, 4),
        num_proc=1,
        file_suffix="buckeye",
        percent_female=percent_f,
    )
    sampled = buckeye_preprocessor._filter_train_dataset(mock_buckeye)
    assert len(sampled.filter(lambda x: x["speaker_gender"] == "f")) == expected_f
    assert len(sampled.filter(lambda x: x["speaker_gender"] == "m")) == expected_m
