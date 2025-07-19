#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_MODULES_CACHE=$HOME/.cache/huggingface/modules

echo "============================================================"
echo "LSTM-CRF中文NER训练脚本"
echo "============================================================"

# 显示中文支持的信息
echo "支持的中文数据集和配置："
PYTHONPATH=. python -c "
from src.utils.lang_utils import get_language_config, get_supported_datasets_for_language
config = get_language_config('zh')
print(f'语言: {config[\"name\"]}')
print(f'分词器: {config[\"tokenizer\"]}')
print(f'BERT模型: {config[\"bert_model\"]}')
print(f'支持的数据集: {get_supported_datasets_for_language(\"zh\")}')
"

echo ""
echo "开始LSTM-CRF中文NER训练..."

# 1. 中文wikiann数据集 - LSTM-CRF模型
echo ""
echo "1. 训练中文wikiann数据集 (LSTM-CRF模型)"
echo "数据集: wikiann (中文)"
echo "模型: LSTM-CRF"
echo "分词: jieba中文分词"
echo "配置: chinese_v4"

PYTHONPATH=. python -W ignore::FutureWarning scripts/train.py \
  --model_type lstm-crf \
  --dataset wikiann \
  --lang zh \
  --num_epochs 30 \
  --batch_size 256 \
  --max_len 128 \
  --learning_rate 0.001 \
  --weight_decay 0.0 \
  --embedding_dim 100 \
  --hidden_dim 256 \
  --log_dir ./logs/lstm_zh \
  --processed_data_dir ./processed_data

# 2. 中文conll2012_ontonotesv5数据集 - LSTM-CRF模型
echo ""
echo "2. 训练中文conll2012_ontonotesv5数据集 (LSTM-CRF模型)"
echo "数据集: conll2012_ontonotesv5 (中文)"
echo "模型: LSTM-CRF"
echo "分词: jieba中文分词"

PYTHONPATH=. python -W ignore::FutureWarning scripts/train.py \
  --model_type lstm-crf \
  --dataset conll2012_ontonotesv5 \
  --lang zh \
  --num_epochs 30 \
  --batch_size 256 \
  --max_len 128 \
  --learning_rate 0.001 \
  --weight_decay 0.0 \
  --embedding_dim 100 \
  --hidden_dim 256 \
  --log_dir ./logs/lstm_zh \
  --processed_data_dir ./processed_data

echo ""
echo "LSTM-CRF中文NER训练完成！"
echo "日志文件保存在 ./logs/lstm_zh 目录"
echo "处理后的数据缓存保存在 ./processed_data 目录"
echo ""
echo "训练结果文件："
echo "- LSTM-CRF + wikiann (中文): ./logs/lstm_zh/lstm-crf_wikiann_zh_bert-base-cased.log"
echo "- LSTM-CRF + conll2012_ontonotesv5 (中文): ./logs/lstm_zh/lstm-crf_conll2012_ontonotesv5_bert-base-cased.log" 