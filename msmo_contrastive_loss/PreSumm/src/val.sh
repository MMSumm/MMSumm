#!/bin/sh
#SBATCH --job-name=bert_section
#SBATCH -A research
#SBATCH -c 15
#SBATCH -o val.out
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=anshul.padhi@research.iiit.ac.in
#SBATCH -w gnode34

python train.py -task abs -mode validate -test_all -batch_size 10000 -test_batch_size 500 -bert_data_path ../bert_data -log_file test_logs -model_path /scratch/new_msmo -sep_optim true -use_interval true -visible_gpus 0,1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path val_out 

