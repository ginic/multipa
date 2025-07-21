import pytest

from datasets import Dataset


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
        "utterance_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "duration": [0.09, 0.09, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
        "buckey_transcript": ["DH IH S IH Z AH T EH1 S T"] * 10,
        "text": ["this is a test"] * 10,
        "ipa": ["ð ɪ s ɪ z ə t ɛ s t"] * 10,
        "speaker_id": [f"S{i:02d}" for i in range(1, 11)],
        "speaker_gender": ["m"] * 5 + ["f"] * 5,
        "speaker_age_range": ["y", "o"] * 5,
    }
    return Dataset.from_dict(mock_dict)
