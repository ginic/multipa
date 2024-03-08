from argparse import ArgumentParser
import os
from pathlib import Path
import re
import shutil
import time

from datasets import load_dataset, Dataset
from epitran import Epitran
import pandas as pd

from multipa.converter.japanese_to_ipa import Japanese2IPA
from multipa.converter.maltese_to_ipa import Maltese2IPA
from multipa.converter.finnish_to_ipa import Finnish2IPA
from multipa.converter.greek_to_ipa import Greek2IPA
from multipa.converter.tamil_to_ipa import Tamil2IPA
from multipa.converter.english_to_ipa import English2IPA
from multipa.converter.buckeye_to_ipa import buckeye_to_ipa

# Constant corpus identifier options
LIBRISPEECH_KEY = "librispeech"
COMMONVOICE_KEY = "commonvoice"
BUCKEYE_KEY = "buckeye"


def transliterate(sample: dict):
    """Performs transliteration for data from CommonVoice or LibriSpeech. 
    Which corpus is automatically determined using which keys are present in the sample dict.
    """
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

def load_dataset_by_corpus_and_language(corpus, language, cache_dir):
    if corpus==LIBRISPEECH_KEY:
        train = load_dataset("librispeech_asr",
                                split="train.clean.100",
                                cache_dir=cache_dir)
        valid = load_dataset("librispeech_asr",
                                split="validation.clean",
                                cache_dir=cache_dir)
    elif corpus==COMMONVOICE_KEY:
        if language == "ta":
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
    else:
        raise ValueError(f"'{corpus}' is not a valid corpus option.")
        
    return train, valid


def resolve_filepath(basename:str, suffix:str, path_prefix:str):
    """Resolve relative file path to by adding appropriate file suffix and path prefix

    Args:
        basename (str): filename without suffix
        suffix (str): desired file suffix
        path_prefix (str): desired file path to pre-pend
    """
    full_path = path_prefix / basename
    return str(full_path.with_suffix(suffix))


def process_buckeye_subfolder(input_directory:Path, output_dir:Path, split:str):
    """Get IPA transcriptions for Buckeye, then write output in the appropriate HuggingFace audiofolder format in output_dir
    Return the number of utterances processed. 

    Args:
        input_directory (Path): A pre-defined split of the Buckeye corpus containing audio files, transcription_data.txt and orthographic_data.txt
        output_dir (Path): Desired output directory
        split (str): identifies the data split, e.g. 'train', 'test'

    Returns:
        int: Number of files processed
    """
    # Combine orthographic transcriptions, Buckeye and IPA transcriptions
    transcription_file = input_directory / "transcription_data.txt"
    transcriptions_df = pd.read_csv(transcription_file, sep="\t", header=None, names=["utterance_id", "duration", "buckeye_transcript"])
    orthography_file = input_directory / "orthographic_data.txt"
    orthography_df = pd.read_csv(orthography_file, sep="\t", header=None, names=["utterance_id", "duration", "text"]).drop(columns="duration")
    transcriptions_df = transcriptions_df.join(orthography_df.set_index("utterance_id"), on="utterance_id")
    transcriptions_df["ipa"] = transcriptions_df["buckeye_transcript"].apply(buckeye_to_ipa)

    # The file paths need to be relative to the parent folder for Hugging face    
    split_dir = output_dir / split
    split_dir.mkdir(exist_ok=True)    
    audio_suffix = ".wav"
    transcriptions_df["file"] = transcriptions_df["utterance_id"].apply(lambda x: resolve_filepath(x, audio_suffix, split_dir))

    # CSV is complete
    # TODO Is it necessary to output JSON here also? 
    transcriptions_df.to_csv(output_dir / f"{split}.csv", index=False)

    # Copy audio to destination folder
    audio_files = list(input_directory.glob(f"*{audio_suffix}"))
    # Number of audio files should match number of transcriptions
    assert len(audio_files) == len(transcriptions_df)
    for f in audio_files:
        shutil.copy2(f, split_dir)

    return len(audio_files)


def main_cli():
    parser = ArgumentParser(description="Create dataset locally.")

    parser.add_argument("--output_dir", type=Path, default="data_new",
                        help="Specify the output directory in which the preprocessed data will be stored.")
    parser.add_argument("--num_proc", type=int, default=1,
                        help="Specify the number of cores to use for multiprocessing. The default is set to 1 (no multiprocessing).")
    parser.add_argument("--clear_cache", action="store_true",
                        help="Use this option if you want to clear the dataset cache after loading to prevent memory from crashing.")
    parser.add_argument("--cache_dir", type=str, default="~/.cache/huggingface/datasets",
                        help="Specify the cache directory's path if you choose to clear the cache.")

    subparsers = parser.add_subparsers(help="Specify which corpus you'll be using", dest="corpus")

    comm_voice_subparser = subparsers.add_parser(COMMONVOICE_KEY, help="Use the Common Voice corpus version 11 from the Huggingface data repo.")
    comm_voice_subparser.add_argument("-l", "--languages", nargs="+", default=["ja", "pl", "mt", "hu", "fi", "el", "ta"],
                        help="Specify the languages to include in the test dataset.")

    librispeech_subparser = subparsers.add_parser(LIBRISPEECH_KEY, help="Use the Librispeech ASR English corpus from the Huggingface data repo.")

    buckeye_subparser = subparsers.add_parser(BUCKEYE_KEY, help="Use the Buckeye corpus with pre-defined train/test splits in local files. This just turns it into the HuggingFace 'audiofolder' format with IPA transcriptions.")
    buckeye_subparser.add_argument("input_dir", type=Path, help="Input directory containing Buckeye corpus divided in 'Train', 'Test', 'Dev' subfolders")


    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)
    stats_file = "{}/presave_trainvalid_stats.tsv".format(args.output_dir)
    with open(stats_file, "w") as f:
        f.write("lang\ttrain\tvalid\ttest\ttime\n")
    
    # test data split creation
    if args.corpus == BUCKEYE_KEY:
        start = time.time()
        sizes = {}
        for split in ["Train", "Dev", "Test"]:
            input_split = args.input_dir / split
            sizes[split] = process_buckeye_subfolder(input_split, args.output_dir, split)

        print("Buckeye num files per split:", sizes)
        end = time.time()
        duration = end - start
        print(f"Elapsed time for Buckeye: {duration}")
        with open(stats_file, "a") as f:
            f.write("{}\t{}\t{}\t{}\t{}\n".format("buckeye", sizes["Train"], sizes["Dev"], sizes["Test"], duration))
        
        
    elif args.corpus in [COMMONVOICE_KEY, LIBRISPEECH_KEY]:
        for language in args.languages:
            start = time.time()
            train, valid = load_dataset_by_corpus_and_language(args.corpus, language, args.cache_dir)
        
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
                f.write("{}\t{}\t{}\t{}\t{}\n".format(language, len(train), len(valid), 0, duration))

        # Clear cache
        print("Clearing the cache...")
        if args.clear_cache:
            train.cleanup_cache_files()
            valid.cleanup_cache_files()
        print("Cache cleared")


if __name__ == "__main__":
    main_cli()