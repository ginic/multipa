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
    mock_dict = {}
    pass
