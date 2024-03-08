#!/bin/bash
# Installation and download IPA dictionaries

pip install --upgrade pip
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
pip install .
python -m unidic download
curl -O https://raw.githubusercontent.com/menelik3/cmudict-ipa/master/cmudict-0.7b-ipa.txt

