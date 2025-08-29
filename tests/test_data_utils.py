import datasets
import pytest

from multipa.data_utils import (
    clean_text,
    extract_all_chars_ipa,
    extract_whitespace_delimited_symbols,
    BuckeyePreprocessor,
    CommonVoicePreprocessor,
    LibriSpeechPreprocessor,
    SimpleSampler,
    SubsetSampler,
)


@pytest.fixture(scope="session")
def buckeye_preprocessor():
    return BuckeyePreprocessor(
        data_dir=None,
        cache_dir=None,
        train_sampler=SimpleSampler(10, 6),
        val_sampler=SimpleSampler(99, 4),
        num_proc=1,
    )


@pytest.fixture(scope="session")
def common_voice_preprocessor():
    return CommonVoicePreprocessor(
        data_dir=None,
        cache_dir=None,
        train_sampler=SubsetSampler(42, [2, 4], ["en", "pl"]),
        val_sampler=SubsetSampler(99, [1, 1], ["en", "pl"]),
        num_proc=1,
        is_remove_spaces=True,
    )


@pytest.fixture(scope="session")
def librispeech_preprocessor():
    return LibriSpeechPreprocessor(
        data_dir=None,
        cache_dir=None,
        train_sampler=SimpleSampler(10, 6),
        val_sampler=SimpleSampler(99, 4),
        num_proc=1,
    )


@pytest.fixture(scope="session")
def mock_librispeech():
    mock_dict = {
        "audio": [{}] * 5,
        "chapter_id": [1234] * 5,
        "id": ["9876"] * 5,
        "speaker_id": [9876] * 5,
        "text": ["THIS IS A TEST"] * 5,
        "ipa": ["ðɪs ɪz ə tɛst"] * 5,
    }
    return datasets.Dataset.from_dict(mock_dict)


@pytest.fixture(scope="session")
def mock_common_voice():
    mock_dict = {
        "audio": [{}] * 10,
        "locale": ["en"] * 5 + ["pl"] * 5,
        "sentence": ["This is a test"] * 5 + ["żaba"] * 5,
        "ipa": ["ðɪs ɪz ə tɛst"] * 5 + ["ʐ a b a"] * 5,
        "age": ["teens", "twenties", "thirties", "forties", "fifties"] * 2,
        "gender": ["male", "female", "male", "female", "male"] * 2,
        "down_votes": [0, 0, 0, 4, 5] * 2,
        "up_votes": [1, 3, 2, 0, 1] * 2,
    }
    return datasets.Dataset.from_dict(mock_dict)


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
    return datasets.Dataset.from_dict(mock_dict)


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
        percent_female=percent_f,
    )
    sampled = buckeye_preprocessor._filter_train_dataset(mock_buckeye)
    assert len(sampled.filter(lambda x: x["speaker_gender"] == "f")) == expected_f
    assert len(sampled.filter(lambda x: x["speaker_gender"] == "m")) == expected_m
    training_stats = buckeye_preprocessor.get_latest_training_dataset_stats()
    assert training_stats["train_num_female_examples"] == expected_f
    assert training_stats["train_num_male_examples"] == expected_m


def test_buckeye_build_vocab(mock_buckeye, buckeye_preprocessor):
    # all symbols from the dataset are in the vocab already
    vocab = buckeye_preprocessor.create_vocabulary(mock_buckeye)
    # There are 62 symbols in the vocab from buckeye_ipa_inventory.txt plus padding and unknown
    assert len(vocab) == 65
    assert vocab["[UNK]"] == 63
    assert vocab["[PAD]"] == 64
    assert " " not in vocab


def test_buckeye_build_vocab_extra_symbol(buckeye_preprocessor):
    # There two new symbols (ʐ, a) in the training data. Do they get added to the vocab?
    data = datasets.Dataset.from_dict({"ipa": ["ð ɪ s ɪ z ə t ɛ s t", "ʐ a b a"]})
    vocab = buckeye_preprocessor.create_vocabulary(data)
    assert len(vocab) == 67


def test_buckeye_clean_ipa(buckeye_preprocessor):
    buckeye_dict = {
        "utterance_id": [1, 2, 3, 4],
        "duration": [0.09, 1.1, 2.2, 3.3],
        "buckey_transcript": [
            "DH IH S IH Z AH T EH1 S T",
            "U U",
            None,
            "dh ae tq",
        ],
        "text": [
            "this is a test",
            "VOCNOISE VOCNOISE",
            None,
            "that",
        ],
        "ipa": [
            "ð ɪ s ɪ z ə t ɛ s t",
            "",
            None,
            "ð æ ʔ",
        ],
        "speaker_id": [f"S{i:02d}" for i in range(1, 5)],
        "speaker_gender": ["m"] * 4,
        "speaker_age_range": ["y", "o"] * 2,
    }
    expected_ipa = [
        "ðɪsɪzətɛst",
        "",
        "",
        "ðæʔ",
    ]

    hf_dataset = datasets.Dataset.from_dict(buckeye_dict)
    clean_dataset = buckeye_preprocessor.clean_ipa_transcription(hf_dataset)
    output_dict = clean_dataset.to_dict()
    assert output_dict.keys() == buckeye_dict.keys()
    assert output_dict["ipa"] == expected_ipa


def test_commonvoice_build_vocab(common_voice_preprocessor, mock_common_voice):
    # all symbols from the dataset are in the vocab already
    vocab = common_voice_preprocessor.create_vocabulary(mock_common_voice)
    # There are 292 symbols in the vocab from full_vocab_ipa.txt plus padding and unknown
    assert len(vocab) == 294
    assert vocab["[UNK]"] == 292
    assert vocab["[PAD]"] == 293


def test_librispeech_clean_ipa(librispeech_preprocessor, mock_librispeech):
    output_data = librispeech_preprocessor.clean_ipa_transcription(mock_librispeech)
    clean_ipa = output_data.to_dict()["ipa"]
    # Spaces are NOT removed
    assert clean_ipa == ["ðɪs ɪz ə tɛst"] * 5


def test_librispeech_build_vocab(librispeech_preprocessor, mock_librispeech):
    vocab = librispeech_preprocessor.create_vocabulary(mock_librispeech)
    # There are 292 symbols in the vocab from full_vocab_ipa.txt plus padding and unknown
    # Additionally, spaces are NOT removed, so " " appears in vocab
    assert len(vocab) == 295
    assert vocab["[UNK]"] == 293
    assert vocab["[PAD]"] == 294


def test_commonvoice_clean_ipa(common_voice_preprocessor, mock_common_voice):
    cleaned_data = common_voice_preprocessor.clean_ipa_transcription(mock_common_voice)
    clean_ipa_out = cleaned_data.to_dict()["ipa"]
    expected_cleaned_ipa = ["ðɪsɪzətɛst"] * 5 + ["ʐaba"] * 5
    assert clean_ipa_out == expected_cleaned_ipa


def test_extract_all_chars_ipa():
    batch = {"ipa": ["dʒ ŋ͡m cʼ ɹ̩"]}
    expected_vocab = set(["d", "ʒ", "͡", "ŋ", "m", "c", "ʼ", "ɹ", "̩", " "])
    vocab = extract_all_chars_ipa(batch)
    assert len(vocab["vocab"]) == len(expected_vocab)
    assert set(vocab["vocab"]) == expected_vocab


def text_extract_all_chars_ipa_batched():
    dataset = datasets.Dataset.from_dict({"ipa": ["dʒc", "ŋ͡m ", "cʼ ", "ɹ̩d"]})
    vocab = dataset.map(extract_all_chars_ipa, batched=True, remove_columns=["ipa"])
    expected_vocab = set(["d", "ʒ", "͡", "ŋ", "m", "c", "ʼ", "ɹ", "̩", " "])
    assert len(vocab["vocab"]) == len(expected_vocab)
    assert set(vocab["vocab"]) == expected_vocab


def test_whitespace_delimited_symbols():
    batch = {"ipa": ["dʒ ŋ͡m cʼ\tɹ̩"]}
    expected_vocab = set(["dʒ", "ŋ͡m", "cʼ", "ɹ̩", " ", "\t"])
    vocab = extract_whitespace_delimited_symbols(batch)
    assert len(vocab["vocab"]) == len(expected_vocab)
    assert set(vocab["vocab"]) == expected_vocab


def test_whitespace_delimited_symbols_batched():
    dataset = datasets.Dataset.from_dict({"ipa": ["dʒ c", "ŋ͡m", "cʼ ", "ɹ̩ d", "d\tc"]})
    vocab = dataset.map(extract_whitespace_delimited_symbols, batched=True, remove_columns=["ipa"])
    expected_vocab = set(["d", "dʒ", "ŋ͡m", "c", "cʼ", "ɹ̩", " ", "\t"])
    assert len(vocab["vocab"]) == len(expected_vocab)
    assert set(vocab["vocab"]) == expected_vocab
