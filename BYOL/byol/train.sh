#!/bin/bash
#SBATCH --job-name=byol_train    
#SBATCH --constraint=A100
#SBATCH --time=8-00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=byolvae_train_%A.log 
#SBATCH --exclude compute-0-7
#SBATCH --mem=1500G

pwd; hostname; date

nvidia-smi

echo "--Env activated"
source /your_path/activate byolenv
echo "--Code Running"
python /your_path/train.py
