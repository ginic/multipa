#!/bin/bash
# Installation for non-GPU and download IPA dictionaries

pip install --upgrade pip
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
pip install .[dev,test]
python -m unidic download
