import os
import csv
from phonecodes import phonecodes

print(phonecodes.CODES)


def read_phn_file(phn_path):
    """Read a .PHN file and return a list of TIMIT symbols"""
    phones = []
    with open(phn_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                phones.append(parts[2])
    return phones


def timit_phones_to_ipa(phones):
    """Convert a sequence of TIMIT phones to IPA"""
    phone_string = " ".join(phones)  # join first
    ipa_string = phonecodes.convert(phone_string, "timit", "ipa", "eng")
    return ipa_string


def process_timit_directory(root_dir, output_csv, dataset_name):
    data = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".phn"):
                phn_path = os.path.join(dirpath, filename)

                try:
                    timit_phones = read_phn_file(phn_path)
                    ipa_string = timit_phones_to_ipa(timit_phones)

                    # Get the relative path of the .wav file from the root_dir
                    rel_path = os.path.relpath(phn_path, root_dir)
                    wav_path = os.path.splitext(rel_path)[0] + ".wav"
                    wav_path = "/" + wav_path.replace(os.sep, "/")  # Normalize to Unix-style path

                    # Add dataset name (train/test) at the front
                    wav_path = f"/{dataset_name}{wav_path}"

                    data.append((wav_path, ipa_string))
                except Exception as e:
                    print(f"Error processing {phn_path}: {e}")

    # Sort data alphabetically by filename
    data.sort(key=lambda x: x[0])

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_filename", "ipa_transcription"])
        writer.writerows(data)

    print(f"Done! Wrote {len(data)} sorted entries to {output_csv}")


# Example usage
if __name__ == "__main__":
    root_complete = "/Users/parthbhangla/Desktop/Multipa_Datasets/TIMIT/COMPLETE"

    process_timit_directory(root_complete, "complete_ipa.csv", "COMPLETE")
