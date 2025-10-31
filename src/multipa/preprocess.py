from argparse import ArgumentParser
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
from multipa.converter.buckeye_to_ipa import buckeye_to_ipa, BUCKEYE_INTERRUPT_SYMBOL

from multipa.data_utils import BUCKEYE_KEY, COMMONVOICE_KEY, LIBRISPEECH_KEY


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
    if corpus == LIBRISPEECH_KEY:
        train = load_dataset("librispeech_asr", split="train.clean.100", cache_dir=cache_dir)
        valid = load_dataset("librispeech_asr", split="validation.clean", cache_dir=cache_dir)
    elif corpus == COMMONVOICE_KEY:
        if language == "ta":
            # Tamil dataset is too big and reaches AFS file path limit
            train = load_dataset(
                "mozilla-foundation/common_voice_11_0", language, split="train", streaming=True, cache_dir=cache_dir
            )
            valid = load_dataset(
                "mozilla-foundation/common_voice_11_0",
                language,
                split="validation",
                streaming=True,
                cache_dir=cache_dir,
            )
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
            train = load_dataset("mozilla-foundation/common_voice_11_0", language, split="train", cache_dir=cache_dir)

            valid = load_dataset("mozilla-foundation/common_voice_11_0", language, split="validation", cache_dir=cache_dir)
    else:
        raise ValueError(f"'{corpus}' is not a valid corpus option.")

    return train, valid


def resolve_filepath(basename: str, suffix: str, path_prefix: Path):
    """Resolve relative file path to by adding appropriate file suffix and path prefix

    Args:
        basename (str): filename without suffix
        suffix (str): desired file suffix
        path_prefix (str): desired file path to pre-pend
    """
    full_path = path_prefix / basename
    return str(full_path.with_suffix(suffix))


def process_buckeye_subfolder(
    input_directory: Path,
    output_dir: Path,
    hugging_face_split: str,
    demographics_df=pd.DataFrame,
    is_keep_interrupts: bool = False,
):
    """Get IPA transcriptions for Buckeye, then write output in the appropriate HuggingFace audiofolder format in output_dir
    Return the number of utterances processed.
    Note that Hugging Face is fussy about split names, they have to be one of specific keywords and are case sensitive,
    see https://huggingface.co/docs/hub/datasets-file-names-and-splits#keywords.

    Args:
        input_directory (Path): A pre-defined split of the Buckeye corpus containing audio files,
            transcription_data.txt and orthographic_data.txt
        output_dir (Path): Desired output directory
        hugging_face_split (str): identifies the name of the split for HuggingFace
        is_keep_interrupts (bool): Set to True to keep the interrupt symbol in IPA output. Defaults to False


    Returns:
        int: Number of files processed
    """
    print("Preparing Buckeye split:", hugging_face_split)
    # Combine orthographic transcriptions, Buckeye and IPA transcriptions
    utt_id = "utterance_id"
    speaker_id = "speaker_id"
    transcription_file = input_directory / "transcription_data.txt"
    transcriptions_df = (
        pd.read_csv(transcription_file, sep="\t", header=None, names=[utt_id, "duration", "buckeye_transcript"])
        .dropna()
        .drop_duplicates()
    )
    orthography_file = input_directory / "orthographic_data.txt"
    orthography_df = (
        pd.read_csv(orthography_file, sep="\t", header=None, names=[utt_id, "duration", "text"])
        .drop(columns="duration")
        .dropna()
        .drop_duplicates()
    )
    transcriptions_df = pd.merge(transcriptions_df, orthography_df, how="left", on=utt_id)
    print("Number of transcripts read:", len(transcriptions_df))
    transcriptions_df["ipa"] = transcriptions_df["buckeye_transcript"].apply(lambda x: buckeye_to_ipa(x, is_keep_interrupts))
    # Filter out empty transcriptions (This should only remove rows when is_keep_interrupts=False)
    transcriptions_df = transcriptions_df.loc[(transcriptions_df["ipa"] != "") & (~transcriptions_df["ipa"].str.isspace())]
    print("Number of transcripts after filtering interrupts:", len(transcriptions_df))

    # Join in demographic info
    transcriptions_df[speaker_id] = transcriptions_df[utt_id].apply(lambda x: x[:3].upper())
    transcriptions_df = pd.merge(transcriptions_df, demographics_df, how="left", on=speaker_id)

    # The file paths need to be relative to the parent folder for Hugging face
    output_split_dir = output_dir / hugging_face_split
    output_split_dir.mkdir(exist_ok=True)
    audio_suffix = ".wav"
    transcriptions_df["file_path"] = transcriptions_df[utt_id].apply(
        lambda x: resolve_filepath(x, audio_suffix, output_split_dir)
    )
    transcriptions_df["file_name"] = transcriptions_df[utt_id].apply(lambda x: x + audio_suffix)

    # CSV is complete
    transcriptions_df.to_csv(output_split_dir / "metadata.csv", index=False)

    # Copy audio to destination folder
    for f in transcriptions_df["file_name"]:
        shutil.copy2(input_directory / f, output_split_dir)

    # Number of audio files should match number of transcriptions
    num_files_copied = len(list(output_split_dir.glob(f"*{audio_suffix}")))
    print("Number of files copied:", num_files_copied)
    assert num_files_copied == len(transcriptions_df)
    return len(transcriptions_df)


def main_cli():
    parser = ArgumentParser(description="Create dataset locally.")

    parser.add_argument(
        "--output_dir",
        type=Path,
        default="data_new",
        help="Specify the output directory in which the preprocessed data will be stored.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Specify the number of cores to use for multiprocessing. The default is set to 1 (no multiprocessing).",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Use this option if you want to clear the dataset cache after loading to prevent memory from crashing.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/.cache/huggingface/datasets",
        help="Specify the cache directory's path if you choose to clear the cache.",
    )

    subparsers = parser.add_subparsers(help="Specify which corpus you'll be using", dest="corpus")

    comm_voice_subparser = subparsers.add_parser(
        COMMONVOICE_KEY, help="Use the Common Voice corpus version 11 from the Huggingface data repo."
    )
    comm_voice_subparser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        default=["ja", "pl", "mt", "hu", "fi", "el", "ta"],
        help="Specify the languages to include in the test dataset.",
    )

    librispeech_subparser = subparsers.add_parser(  # noqa: F841
        LIBRISPEECH_KEY, help="Use the Librispeech ASR English corpus from the Huggingface data repo."
    )

    buckeye_subparser = subparsers.add_parser(
        BUCKEYE_KEY,
        help=(
            "Use the Buckeye corpus with pre-defined train/test splits in local files. This just turns it "
            "into the HuggingFace 'audiofolder' format with IPA transcriptions."
        ),
    )
    buckeye_subparser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing Buckeye corpus divided in 'Train', 'Test', 'Dev' subfolders",
    )
    buckeye_subparser.add_argument(
        "--keep_interrupts",
        action="store_true",
        help=(
            f"Use this flag if you want to keep the interrupt symbol '{BUCKEYE_INTERRUPT_SYMBOL}' in "
            "transcripts when converting to IPA"
        ),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)
    stats_file = args.output_dir / "presave_trainvalid_stats.tsv"
    with open(stats_file, "w") as f:
        f.write("lang\ttrain\tvalid\ttest\ttime\n")

    # test data split creation
    if args.corpus == BUCKEYE_KEY:
        start = time.time()
        sizes = {}
        demographics_df = pd.read_csv(
            args.input_dir / "speaker_demos.txt",
            sep=" ",
            names=["speaker_id", "speaker_gender", "speaker_age_range", "interviewer_gender"],
        )
        demographics_df["speaker_id"] = demographics_df["speaker_id"].astype(str)
        for original_split, huggingface_split in [("Train", "train"), ("Dev", "validation"), ("Test", "test")]:
            input_split = args.input_dir / original_split
            sizes[huggingface_split] = process_buckeye_subfolder(
                input_split, args.output_dir, huggingface_split, demographics_df, args.keep_interrupts
            )

        print("Buckeye num files per split:", sizes)
        end = time.time()
        duration = end - start
        print(f"Elapsed time for Buckeye: {duration}")
        with open(stats_file, "a") as f:
            f.write("{}\t{}\t{}\t{}\t{}\n".format("buckeye", sizes["train"], sizes["validation"], sizes["test"], duration))

    elif args.corpus in [COMMONVOICE_KEY, LIBRISPEECH_KEY]:
        if args.corpus == COMMONVOICE_KEY:
            languages = args.languages
        else:
            # Hard code Librispeech as English
            languages = ["en"]

        for language in languages:
            start = time.time()
            train, valid = load_dataset_by_corpus_and_language(args.corpus, language, args.cache_dir)

            # Remove audio column (non-writable to json)
            train, valid = remove_audio_column(train, valid)

            train = train.map(transliterate, num_proc=args.num_proc)
            valid = valid.map(transliterate, num_proc=args.num_proc)

            # Export to json
            train.to_json(args.output_dir / f"{language}_train.json")
            valid.to_json(args.output_dir / f"{language}_valid.json")

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
