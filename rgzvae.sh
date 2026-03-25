#!/bin/bash
#SBATCH --job-name=RGZVAE
#SBATCH --constraint=A100
#SBATCH --time=8-00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2_3/jalphonse/VAE_RadioGalaxies/outs/out_%j_mbkfold.out
#SBATCH --exclude compute-0-7

pwd;

nvidia-smi
echo "--Env activated"
source /share/nas2_3/jalphonse/anaconda3/bin/activate myenv
echo "--Code Running"
python /share/nas2_3/jalphonse/VAE_RadioGalaxies/VAE_RGZ_mb16.py
