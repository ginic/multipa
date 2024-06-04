#!/bin/bash
# Installation and download IPA dictionaries

pip install --upgrade pip
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
pip install .[dev,test]
python -m unidic download
