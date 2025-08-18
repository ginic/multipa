import pytest

from multipa.preprocess import transliterate


@pytest.mark.parametrize(
    "sample,expected_ipa",
    [
        # CommonVoice
        ({"sentence": "this is a test", "locale": "en"}, "ðɪsɪzətɛst"),
        # Librispeech
        ({"text": "this is a test", "chapter_id": 123}, "ðɪsɪzətɛst"),
    ],
)
def test_transliterate(sample, expected_ipa):
    expected = {"ipa": expected_ipa}
    expected.update(sample)
    output = transliterate(sample)
    assert expected == output
