#!/bin/bash
#SBATCH --job-name=byoljj    
#SBATCH --constraint=A100
#SBATCH --time=8-00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=byolvae-evalfttest_%A_norot.log 
#SBATCH --exclude compute-0-7
#SBATCH --mem=1500G

pwd; hostname; date

nvidia-smi

echo "--Env activated"
source /share/nas2_3/jalphonse/anaconda3/bin/activate byolenv
echo "--Code Running"
python /share/nas2_3/jalphonse/BYOL/byol/byol/run_eval1.py