from argparse import ArgumentParser
import os
from pathlib import Path
import re
import time

from datasets import load_dataset, Dataset
from epitran import Epitran
import pandas as pd

import sys
sys.path.insert(0, "./converter")

from japanese_to_ipa import Japanese2IPA
from maltese_to_ipa import Maltese2IPA
from finnish_to_ipa import Finnish2IPA
from greek_to_ipa import Greek2IPA
from tamil_to_ipa import Tamil2IPA
from english_to_ipa import English2IPA

parser = ArgumentParser(description="Create dataset locally.")


parser.add_argument("--output_dir", type=str, default="data_new",
                    help="Specify the output directory in which the preprocessed data will be stored.")
parser.add_argument("--num_proc", type=int, default=1,
                    help="Specify the number of cores to use for multiprocessing. The default is set to 1 (no multiprocessing).")
parser.add_argument("--clear_cache", action="store_true",
                    help="Use this option if you want to clear the dataset cache after loading to prevent memory from crashing.")
parser.add_argument("--cache_dir", type=str, default="~/.cache/huggingface/datasets",
                    help="Specify the cache directory's path if you choose to clear the cache.")

subparsers = parser.add_subparsers(help="Specify which corpus you'll be using", dest="corpus")

comm_voice_subparser = subparsers.add_parser("commonvoice", help="Use the Common Voice corpus version 11 from the Huggingface data repo.")
comm_voice_subparser.add_argument("-l", "--languages", nargs="+", default=["ja", "pl", "mt", "hu", "fi", "el", "ta"],
                    help="Specify the languages to include in the test dataset.")

librispeech_subparser = subparsers.add_parser("librispeech", help="Use the Librispeech ASR English corpus from the Huggingface data repo.")

buckeye_subparser = subparsers.add_parser("buckeye", help="Use the Buckeye corpus with pre-defined train/test splits in local files.")
parser.add_argument("--train_dir","-r", type=Path, help="Directory containing train data split for Buckeye")
parser.add_argument("--dev_dir","-d", type=Path, help="Directory containing validation/dev data split for Buckeye")
parser.add_argument("--test_dir","-e", type=Path, help="Directory containing test data split for Buckeye")


args = parser.parse_args()


def transliterate(sample: dict):
    # Dataset is librispeech_asr
    if "chapter_id" in sample:
        lang = "en"
        sent_key = "text"

    # Dataset is common voice
    else:
        lang = sample["locale"]
        sent_key = "sentence"
    sent = sample[sent_key]
    if lang == "ja":
        converter = Japanese2IPA()
        ipa = converter.remove_ja_punct(sent)
        ipa = converter.convert_sentence_to_ipa(ipa)
    elif lang == "mt":
        ipa = Maltese2IPA().maltese_generate_ipa(sent)
    elif lang == "fi":
        ipa = Finnish2IPA().finnish_generate_ipa(sent)
    elif lang == "el":
        ipa = Greek2IPA().greek_generate_ipa(sent)
    elif lang == "hu":
        ipa = re.findall(r"[\s\w]", sent.lower(), re.MULTILINE)
        ipa = "".join(ipa)
        epi = Epitran("hun-Latn")
        ipa = epi.transliterate(ipa)
    elif lang == "pl":
        ipa = re.findall(r"[\s\w]", sent.lower(), re.MULTILINE)
        ipa = "".join(ipa)
        epi = Epitran("pol-Latn")
        ipa = epi.transliterate(ipa)
    elif lang == "ta":
        ipa = Tamil2IPA().tamil_generate_ipa(sent)
    elif lang == "en":
        ipa = English2IPA().english_generate_ipa(sent)
    else:
        raise Exception("Unknown locale (language) found")
    sample["ipa"] = "".join(ipa.split())
    return sample

def remove_tamil_special_char(train, valid) -> tuple:
    """Remove sentences including "ச" since its pronunciation
    seems to be irregular/erratic
    """
    train = train.filter(lambda batch: "ச" not in batch["sentence"])
    valid = valid.filter(lambda batch: "ச" not in batch["sentence"])
    return train, valid

def remove_audio_column(train, valid) -> tuple:
    """Remove ["audio"] column from the dataset so that it can be
    saved to json.
    Apparently `array` causes `OverflowError: Unsupported UTF-8 sequence length when encoding string`
    so we need to remove it.
    This column will be restored by directly downloaded data upon training.
    """
    train = train.remove_columns(["audio"])
    valid = valid.remove_columns(["audio"])
    return train, valid

def load_dataset_by_language(language, cache_dir):

    if language == "en":
        train = load_dataset("librispeech_asr",
                                split="train.clean.100",
                                cache_dir=cache_dir)
        valid = load_dataset("librispeech_asr",
                                split="validation.clean",
                                cache_dir=cache_dir)
    elif language == "ta":
        # Tamil dataset is too big and reaches AFS file path limit
        train = load_dataset("mozilla-foundation/common_voice_11_0", language,
                                split="train",
                                streaming=True,
                                cache_dir=cache_dir)
        valid = load_dataset("mozilla-foundation/common_voice_11_0", language,
                                split="validation",
                                streaming=True, 
                                cache_dir=cache_dir)
        ds_train = []
        ds_valid = []
        for i, batch in enumerate(train):
            if i >= 30000:
                break
            ds_train.append(batch)
        for i, batch in enumerate(valid):
            if i >= 30000:
                break
            ds_valid.append(batch)
        train = Dataset.from_pandas(pd.DataFrame(data=ds_train))
        valid = Dataset.from_pandas(pd.DataFrame(data=ds_valid))

        train, valid = remove_tamil_special_char(train, valid)
        
    else:
        train = load_dataset("mozilla-foundation/common_voice_11_0", language,
                                split="train", cache_dir=cache_dir)
                        
        valid = load_dataset("mozilla-foundation/common_voice_11_0", language,
                                split="validation", cache_dir=cache_dir)
        
    return train, valid

# Dataset
if __name__ == "__main__":
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    stats_file = "{}/presave_trainvalid_stats.tsv".format(args.output_dir)
    with open(stats_file, "w") as f:
        f.write("lang\ttrain\tvalid\ttime\n")
    start = time.time()        
    # TODO Check for different corpus here, then only Common Voice needs to deal with multiple languages. 

    # test data split creation
    for language in args.languages:
        train, valid = load_dataset_by_language(language, args.cache_dir)
    
        # Remove audio column (non-writable to json)
        train, valid = remove_audio_column(train, valid)

        train = train.map(transliterate,
                          num_proc=args.num_proc)
        valid = valid.map(transliterate,
                          num_proc=args.num_proc)

        # Export to json
        train.to_json("{}/{}_train.json".format(args.output_dir, language))
        valid.to_json("{}/{}_valid.json".format(args.output_dir, language))

        print("{}\ttrain: {}\tvalid: {}\n".format(language, len(train), len(valid)))
        end = time.time()
        duration = end - start
        print("Elapsed time for {}: {}".format(language, duration))
        with open(stats_file, "a") as f:
            f.write("{}\t{}\t{}\t{}\n".format(language, len(train), len(valid), duration))

        # Clear cache
        print("Clearing the cache...")
        if args.clear_cache:
            train.cleanup_cache_files()
            valid.cleanup_cache_files()
        print("Cache cleared")
