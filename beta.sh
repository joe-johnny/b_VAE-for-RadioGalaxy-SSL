#!/bin/bash

#SBATCH --job-name=beta_search_$RUN_ID
#SBATCH --constraint=A100
#SBATCH --time=3-00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/'your_path'/output_beta/out_%j.out
#SBATCH --exclude compute-0-7

pwd;

nvidia-smi
echo "--Env activated"
source /'your_path'/anaconda3/bin/activate myenv
echo "--Code Running"
python /'your_path'/beta_search.py
