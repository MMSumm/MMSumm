python train.py  -task abs -mode train -bert_data_path /scratch/full_data_pt/ -dec_dropout 0.2 \
-model_path /scratch/new_msmo -sep_optim true -lr_dec 0.02 -save_checkpoint_steps \
2000 -batch_size 300 -train_steps 200000 -report_every 1 -accum_count 5 -use_bert_emb true \
-use_interval true -warmup_steps_bert 16000 -warmup_steps_dec 8000 -max_pos 512 -visible_gpus \
0 -log_file log
