from datasets import Dataset

import multipa.main as multipa_train


def test_create_vocabulary():
    df1 = {"ipa":["abc", "ðɪs"]}
    df2 = {"ipa":["tɛst"]}
    expected_vocab = set("abcðɪstɛst") | set(["[UNK]", "[PAD]"])
    vocab = multipa_train.create_vocabulary(Dataset.from_dict(df1), Dataset.from_dict(df2), use_resource_vocab=False)
    assert expected_vocab == set(vocab.keys())


def test_create_vocabulary_default_vocab():
    vocab = multipa_train.create_vocabulary()
    assert len(vocab) == 294
    # Last two elements in vocab should be unknown and padding
    assert vocab["[UNK]"] == 292
    assert vocab["[PAD]"] == 293