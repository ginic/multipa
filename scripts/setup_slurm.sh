#!/bin/bash
# Installation and download IPA dictionaries

#SBATCH -c 4
#SBATCH --mem=12GB
#SBATCH -G 2
#SBATCH -p gpu-preempt
#SBATCH --time 02:00:00
#SBATCH -o setup_%j.out
#SBATCH --mail-type END

module load miniconda/22.11.1-1
conda env create --prefix ./env --file=multipa.yml
conda activate ./env

python --version

echo "How many GPUs found by pytorch?"
python -c "import torch; print(torch.cuda.device_count())"

pip install .[gpu,dev,test]

python -m unidic download

python -m pytest