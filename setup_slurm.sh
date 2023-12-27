#!/bin/bash
# Installation and download IPA dictionaries

#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH -p gpu-preempt
#SBATCH --time 01:00:00
#SBACTH -o setup_%j.out
#SBATCH --mail-type END

module load miniconda/22.11.1-1
module load cuda/11.3.1

conda create -n multipa python=3.10 -y
conda activate multipa

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install --upgrade pip

pip install -r requirements.txt
python -m unidic download
curl -O https://raw.githubusercontent.com/menelik3/cmudict-ipa/master/cmudict-0.7b-ipa.txt
