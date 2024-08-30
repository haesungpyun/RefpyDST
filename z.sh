#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate refpydst

hour=$(date +"%H")


# CUDA_VISIBLE_DEVICES=0 nohup python run_sampling_exp_vllm.py runs/table4_llama/5p/smapling_exp_train/split_v1_topk_bm_5_fs_5_8B_vllm_remain_split_2.json &> logs/llama_vllm_train_set_sampling_exp_topk_bm_5_fs_5_8B_2.out&
# # get pid of the last comman
# pid_0=$!

# CUDA_VISIBLE_DEVICES=1 nohup python run_sampling_exp_vllm.py runs/table4_llama/5p/smapling_exp_train/split_v1_topk_bm_5_fs_5_8B_vllm_remain_split_4.json &> logs/llama_vllm_train_set_sampling_exp_topk_bm_5_fs_5_8B_4.out&
# pid_1=$!

# echo "Started the experiments for topk_bm_5_fs_5_8B 5 and 3"

while [ $hour -lt 9 ]; do
    echo "Waiting to kill... Current Hour: $hour"
    echo "Sleeping for 30m"
    sleep 30m
    hour=$(date +"%H")
done
kill -9 $pid_1
kill -9 $pid_0

# while [ $hour -lt 21 ]; do
#     echo "Waiting to kill... Current Hour: $hour"
#     echo "Sleeping for 30m"
#     sleep 30m
#     hour=$(date +"%H")
# done
# kill -9 $pid
