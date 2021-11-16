#!/bin/sh
#SBATCH --job-name=bert_section
#SBATCH -A research
#SBATCH -c 30
#SBATCH -o bert_section.out
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=anshul.padhi@research.iiit.ac.in
#SBATCH -w gnode33

# # # old = lr_dec = 0.2, lr_bert = 0.002, warmup steps bert 20k warmup steps dec 10k

python train.py  -task abs -mode train -bert_data_path /scratch/full_data_pt/ -dec_dropout 0.2 \
-model_path /scratch/new_msmo -sep_optim true -lr_dec 0.02 -save_checkpoint_steps \
2000 -batch_size 3000 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true \
-use_interval true -warmup_steps_bert 16000 -warmup_steps_dec 8000 -max_pos 512 -visible_gpus \
0,1,2,3 -log_file log

#### CHANGED GPUS REMEMBER GANDU, REDUCED BATCH SIZE TOO
