#!/bin/bash
# Installation and download IPA dictionaries

#SBATCH -c 4
#SBATCH --mem=12GB
#SBATCH -G 2
#SBATCH -p gpu-preempt
#SBATCH --time 02:00:00
#SBATCH -o setup_%j.out
#SBATCH --mail-type END

module load conda/latest
conda env create --prefix ./env_cuda124 --file=multipa_cuda124.yml
conda activate ./env_cuda124

conda list

python --version

pip install .[gpu,dev,test]

conda list

echo "How many GPUs found by pytorch?"
python -c "import torch; print(torch.cuda.device_count())"

python -m unidic download

python -m pytest
