#!/bin/bash
#SBATCH -n 9
#SBATCH -w gnode38
#SBATCH --mem-per-cpu=2048
#SBATCH --time=96:00:00
#SBATCH --mincpus=9
#SBATCH --gres=gpu:1
#SBATCH --mail-user=anshul.padhi@ada.iiit.ac.in
#SBATCH --mail-type=ALL

python3 convert_to_dict.py
