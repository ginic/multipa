import pytest

from multipa.converter.english_to_ipa import English2IPA
from multipa.converter.buckeye_to_ipa import buckeye_to_ipa


def test_english_generate_ipa():
    default_eng_converter = English2IPA(keep_suprasegmental=False)
    expected = "ðɪs ɪz ə tɛst"
    output = default_eng_converter.english_generate_ipa("this is a test")
    assert expected == output


@pytest.mark.parametrize(
    "buckeye_input,expected_ipa,keep_interrupts",
    [
        ("U ah m", "U ʌ m", True),
        ("U ah m", "ʌ m", False),
        ("ah m U", "ʌ m U", True),
        ("ah m U", "ʌ m", False),
        ("dh ih s ih z U ah t eh s t", "ð ɪ s ɪ z ʌ t ɛ s t", False),
        ("dh ih s ih z U ah t eh s t", "ð ɪ s ɪ z U ʌ t ɛ s t", True),
        ("dh ih s ih z ah t eh s t", "ð ɪ s ɪ z ʌ t ɛ s t", False),
        ("U U", "", False),
        ("U", "", False),
        ("dh s VOCNOISE k", "ð s U k", True),
        ("dh s UNKNOWN k", "ð s k", False),
        ("dh s SIL k", "ð s k", False),
        ("dh s SIL k", "ð s U k", True),
    ],
)
def test_buckeye_to_ipa(buckeye_input, expected_ipa, keep_interrupts):
    actual = buckeye_to_ipa(buckeye_input, keep_interrupts)
    assert actual == expected_ipa
