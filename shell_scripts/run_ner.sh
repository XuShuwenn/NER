#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7,8
export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_MODULES_CACHE=$HOME/.cache/huggingface/modules

# 模型和数据集列表
models=("bert" "lstm-crf")
datasets=("conll2003" "conll2012_ontonotesv5" "wikiann")

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    if [ "$dataset" = "conll2012_ontonotesv5" ]; then
      lang="english_v4"
    else
      lang="en"
    fi
    echo "=============================="
    echo "Running model: $model, dataset: $dataset"
    echo "=============================="
    if [ "$model" = "bert" ]; then
      PYTHONPATH=. python -W ignore::FutureWarning scripts/train.py \
        --model_type bert \
        --dataset "$dataset" \
        --lang $lang \
        --model_name_or_path bert-base-cased \
        --num_epochs 20 \
        --batch_size 128 \
        --max_len 128 \
        --learning_rate 4e-5 \
        --weight_decay 0.01 \
        --log_dir ./logs \
        --processed_data_dir ./processed_data
    else
      PYTHONPATH=. python -W ignore::FutureWarning scripts/train.py \
        --model_type lstm-crf \
        --dataset "$dataset" \
        --lang $lang \
        --num_epochs 10 \
        --batch_size 256 \
        --max_len 128 \
        --learning_rate 0.001 \
        --weight_decay 0.0 \
        --embedding_dim 100 \
        --hidden_dim 256 \
        --log_dir ./logs \
        --processed_data_dir ./processed_data
        # --embedding_path /path/to/embeddings.txt  # 如有预训练词向量可加
    fi
  done
done