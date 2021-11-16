#!/bin/bash
#SBATCH -w gnode36
#SBATCH --job-name=bert_section
#SBATCH -A research
#SBATCH -c 10
#SBATCH -o bert_section.out
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END


python3 simplify.py
