from multipa.converter.english_to_ipa import English2IPA

def test_english_generate_ipa():
    default_eng_converter = English2IPA(keep_suprasegmental=False)
    expected = "ðɪs ɪz ə tɛst"
    output = default_eng_converter.english_generate_ipa("this is a test")
    assert expected == output