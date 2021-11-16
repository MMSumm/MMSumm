#!/bin/bash
#SBATCH -n 5
#SBATCH -w gnode36
#SBATCH --mem-per-cpu=2048
#SBATCH --time=96:00:00
#SBATCH --mincpus=10
#SBATCH --gres=gpu:2
#SBATCH --mail-user=tanmay.sachan@research.iiit.ac.in
#SBATCH --mail-type=ALL

python get_vgg_embeddings.py
