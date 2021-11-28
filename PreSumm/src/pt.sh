#!/bin/sh
#SBATCH --job-name=preproc_data
#SBATCH -A research
#SBATCH -c 4
#SBATCH -o preproc.out
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=tanmay.sachan@research.iiit.ac.in
#SBATCH -w gnode38


python preprocess.py -mode format_to_bert -raw_path ../json_data -save_path ../bert_data -lower
-n_cpus 30 -log_file logs
