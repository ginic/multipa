# TODO This assumes a path structure which might not exist. Make the path configurable input by the user.
import os
import glob
from datasets import Dataset, Audio, concatenate_datasets
import pandas as pd
from typing import List


def add_language() -> List[dict]:
    # a path looks like this: "additional/Xhosa/Audio/{$AUDIOFILE}.wav" or "additional/Xhosa/{$PRONFILE}.txt
    path = "./additional/"
    if not os.path.exists(path):
        path = "../additional/"
    langs = [lang for lang in os.listdir(path) if not lang.startswith(".")]
    for lang in langs:
        assert lang.istitle(), print("The first letter of the language must be upper-case (i.e., title case)")

        audio_path = path + "/" + lang + "/Audio/wav"
        text_file = path + "/" + lang + "/ipa.txt"
        audio_files = os.listdir(audio_path)  # -> list of files in the directory

        with open(text_file, "r") as f:
            # a line will look like this:
            # pronunciation_xh_amabhaca amabaːǀa
            ipas = [line.strip() for line in f.readlines() if line != "\n"]
            assert not ipas[0].endswith("\n"), print("line break letter found at the end of the line.")
            assert len(ipas) == len(audio_files), print("Numbers of audio files and IPAs don't match")

        ds = dict()
        ds["path"] = list()
        ds["ipa"] = list()
        ds["sentence"] = list()
        for lang in ipas:
            # print(l.split(" "))
            filename = lang.split()[0]
            pron = " ".join(lang.split()[1:])
            sent = " ".join(filename.split("_")[2:])
            print(filename)
            filename = filename + ".wav"
            assert filename in audio_files, print("Audio file not found. Check the file name or the directory.")
            file_path = audio_path + "/" + filename
            ds["path"].append(file_path)
            ds["ipa"].append(pron)
            ds["sentence"].append(sent)
        df = pd.DataFrame(ds)  # -> DataFrame
        ds = Dataset.from_pandas(df)  # -> Dataset

        # Read binary data (array) of the audio files
        audio_files_with_path = glob.glob(audio_path + "/*")
        audio_data = Dataset.from_dict({"audio": audio_files_with_path}).cast_column("audio", Audio(sampling_rate=48000))
        # -> Dataset

        # Concatenate ds and the audio column w.r.t. the column
        ds = concatenate_datasets([ds, audio_data], axis=1)

    return ds
