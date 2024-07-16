#!/bin/bash
# Installation and download IPA dictionaries

#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH -G 1
#SBATCH -p gpu-preempt
#SBATCH --time 02:00:00
#SBATCH -o setup_%j.out
#SBATCH --mail-type END

module load miniconda/22.11.1-1
module load cuda/11.8.0

conda create -n multipa python=3.9 -y
conda activate multipa

conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 cudatoolkit=11.8 -c pytorch -y

pip install --upgrade pip

pip install .[gpu,dev,test]
python -m unidic download

pytest