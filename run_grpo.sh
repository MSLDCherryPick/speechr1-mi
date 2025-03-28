#!/bin/bash

OUT_DIR=exp/model
MODEL_NP=Qwen/Qwen2-Audio-7B-Instruct
DATA_FILE=data/AVQA/train_qa.data

# GPU_NUM=$(nvidia-smi -L | wc -l)
GPU_NUM=1
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=32777
# export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
torchrun --nproc_per_node=${GPU_NUM} \
    --nnodes=${NODE_NUM} \
    --node-rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train.py \
    --config_path conf/ds_zero3.json \
    --model_name_or_path ${MODEL_NP} \
    --out_dir ${OUT_DIR} \
    --data_file ${DATA_FILE} \
    --use_wandb false || exit 1
