#!/bin/bash
#SBATCH --job-name=RGZVAE
#SBATCH --constraint=A100
#SBATCH --time=8-00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/'your_path'/out_%j_mbkfold.out
#SBATCH --exclude compute-0-7

pwd;

nvidia-smi
echo "--Env activated"
source /'your_path'/anaconda3/bin/activate myenv
echo "--Code Running"
python 'your_path'/VAE_for_RGZData.py
