#!/bin/bash
#SBATCH -n 4
#SBATCH -w gnode36
#SBATCH --mem-per-cpu=2048
#SBATCH --time=96:00:00
#SBATCH --mincpus=5
#SBATCH --gres=gpu:1
#SBATCH --mail-user=tanmay.sachan@research.iiit.ac.in
#SBATCH --mail-type=ALL

python oscarify_data.py 1
