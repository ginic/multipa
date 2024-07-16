#!/bin/bash
# Installation and download IPA dictionaries

pip install --upgrade pip
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
pip install .[dev,test]
python -m unidic download
