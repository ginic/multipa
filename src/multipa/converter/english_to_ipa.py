import importlib.resources

import eng_to_ipa as ipa


def make_prondict(filename: str) -> dict:
    with open(filename, "r") as f:
        entries = f.readlines()
        prondict = dict()
        for e in entries:
            e = e.strip()
            k, v = e.split("\t")
            if "," in v:
                v = v.split(", ")
                prondict[k.lower()] = v
            else:
                prondict[k.lower()] = [v]
    return prondict


class English2IPA:
    def __init__(self, keep_suprasegmental=False, filename=None):
        """Generate english

        Args:
            keep_suprasegmental (bool, optional): Set to true to keep stress markers. Defaults to False.
            filename (str, optional): Path to ipa dictionary filename. Defaults to "cmudict-0.7b-ipa.txt".
        """
        if filename is None:
            self.prondict = make_prondict(importlib.resources.files("multipa.resources").joinpath("cmudict-0.7b-ipa.txt"))
        else:
            self.prondict = make_prondict(filename)
        self.keep_suprasegmental = keep_suprasegmental

    def english_generate_ipa(self, sent: str):
        words = sent.lower().split()
        transcription = []
        keys = self.prondict.keys()
        for w in words:
            if w not in keys:
                addendum = ipa.convert(w)
            else:
                addendum = self.prondict[w][0]
            if not self.keep_suprasegmental:
                addendum = addendum.replace("ˈ", "").replace("ˌ", "")
            transcription.append(addendum)
        output = " ".join(transcription)
        return output
