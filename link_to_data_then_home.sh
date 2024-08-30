#!/bin/bash


mkdir -p /data1/home/$USER/.cache/huggingface/hub/
mkdir -p /home/$USER/.cache/huggingface/hub/

data_cache_dir=/data1/home/$USER/.cache/huggingface/hub/
home_cache_dir=/home/$USER/.cache/huggingface/hub/

# /data1/home/$USER/.cache로 링킹 이후 그것을 다시 /home/$USER/.cache로 링킹
# /data1/meta-llama/.cache/huggingface/hub/ 아래에 있는 모든 폴더를 순회
for dir in /data1/meta-llama/.cache/huggingface/hub/*; do
    model_name=$(basename "$dir")
    data_cache_model="$data_cache_dir$model_name"
    home_cache_model="$home_cache_dir$model_name"
    
    # if the data_cache_model is directory, then continue
    # if the data_cache_model is symbolic link, then unlink it
    if [ -d "$data_cache_model" ]; then
        echo "$data_cache_model exists"
        continue
    fi
    if [ -L "$home_cache_model" ]; then
        unlink $data_cache_model
    fi

    cd $data_cache_dir
    ln -s $dir .
    echo "linked /data1/meta-llama/.cache/.../$model_name to data1/$USER/.cache/.../$model_name"

    # if the home_cache_model is directory then continue
    # if the home_cache_model is symbolic link, then unlink it
    if [ -d "$home_cache_model" ]; then
        echo "$home_cache_model exists"
        continue
    fi
    if [ -L "$home_cache_model" ]; then
        unlink $home_cache_model
    fi

    cd $data_cache_dir
    ln -s $data_cache_model .
    echo "linked data1/$USER/.cache/.../$model_name to /home/$USER/.cache/.../$model_name"

done

echo "done"