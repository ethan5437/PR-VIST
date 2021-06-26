#!/bin/bash
set -x
CUDA_LAUNCH_BLOCKING=1 python3 ../src/main.py --model HR_BiLSTM_plus --dataset vist --framework UHop --earlystop_tolerance 20 --saved_dir saved_model --dynamic none --reduce_method dense --epoch_num 200 --emb_size 300 --hidden_size 150 --dropout_rate 0.2 --learning_rate 0.0001 --optimizer rmsprop --l2_norm 0.0 --margin 0.5 --hop_weight 1 --task_weight 1 --acc_weight 1 --q_representation lstm --occurrence 2 --termset_sample_size 1000 --sample_size 150 --device 0 --path_search --step_every_step --train --is_image_abs_position --relation_frequency 10 
