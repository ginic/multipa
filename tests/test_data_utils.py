import datasets

from multipa.data_utils import clean_text

def test_clean_text():
    examples = [{"text":"hello", "ipa":"hɛlo"}, 
                {"text": "U U", "ipa": None}, 
                {"text": "this is a test", "ipa": "ðɪs ɪz ə tɛst"}]
    input_dataset = datasets.Dataset.from_list(examples)
    output_dataset = input_dataset.map(lambda x: clean_text(x, is_remove_space=True))
    assert len(output_dataset) == 3
    assert {"text":"hello", "ipa":"hɛlo"} in output_dataset
    assert {"text": "U U", "ipa":""} in output_dataset
    assert {"text": "this is a test", "ipa": "ðɪsɪzətɛst"} in output_dataset
