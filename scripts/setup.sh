#!/bin/bash
# Installation for non-GPU and download IPA dictionaries

pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install .[dev,test]
python -m unidic download
