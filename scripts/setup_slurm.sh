#!/bin/bash
# Installation and download IPA dictionaries

#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH -G 1
#SBATCH -p gpu-preempt
#SBATCH --time 02:00:00
#SBATCH -o setup_%j.out
#SBATCH --mail-type END

module load python/3.11.0
module load cuda/11.8.0

python -m venv venv
source venv/bin/activate

pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyarrow levenshtein
echo "How many GPUs found by pytorch?"
python -c "import torch; print(torch.cuda.device_count())"

pip install .[gpu,dev,test]

python -m unidic download

pytest