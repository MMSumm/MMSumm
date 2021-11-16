#!/bin/sh
#SBATCH --job-name=bert_section
#SBATCH -A research
#SBATCH -c 15
#SBATCH -o test.out
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=anshul.padhi@research.iiit.ac.in
#SBATCH -w gnode33

python train.py -task abs -mode test -test_from /scratch/new_msmo/model_step_26000.pt -batch_size 140 -test_batch_size 50 -bert_data_path ../bert_data/ -log_file test_log -sep_optim true -use_interval true -visible_gpus 0,1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path test


## REMEMBER GPUS LESS
