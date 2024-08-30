#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate refpydst

hour=$(date +"%H")


CUDA_VISIBLE_DEVICES=2 nohup python run_sampling_exp_vllm.py runs/table4_llama/5p/smapling_exp_train/split_v1_topk_bm_5_fs_5_8B_vllm_remain_split_5.json &> logs/llama_vllm_train_set_sampling_exp_topk_bm_5_fs_5_8B_5.out&
# get pid of the last comman
pid=$!
wait $pid

CUDA_VISIBLE_DEVICES=2 nohup python run_sampling_exp_vllm.py runs/table4_llama/5p/smapling_exp_train/split_v1_topk_bm_5_fs_5_8B_vllm_remain_split_3.json &> logs/llama_vllm_train_set_sampling_exp_topk_bm_5_fs_5_8B_3.out&
# pid_3=$!

# echo "Started the experiments for topk_bm_5_fs_5_8B 5 and 3"

# while [ $hour -lt 12 ]; do
#     echo "Waiting to kill... Current Hour: $hour"
#     echo "Sleeping for 30m"
#     sleep 30m
#     hour=$(date +"%H")
    
# done
# kill -9 $pid_3

# while [ $hour -lt 21 ]; do
#     echo "Waiting to kill... Current Hour: $hour"
#     echo "Sleeping for 30m"
#     sleep 30m
#     hour=$(date +"%H")
# done
# kill -9 $pid
