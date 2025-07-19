#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_MODULES_CACHE=$HOME/.cache/huggingface/modules

echo "============================================================"
echo "LSTM-CRF英文NER训练脚本"
echo "============================================================"

# 显示英文支持的信息
echo "支持的英文数据集和配置："
PYTHONPATH=. python -c "
from src.utils.lang_utils import get_language_config, get_supported_datasets_for_language
config = get_language_config('en')
print(f'语言: {config[\"name\"]}')
print(f'分词器: {config[\"tokenizer\"]}')
print(f'BERT模型: {config[\"bert_model\"]}')
print(f'支持的数据集: {get_supported_datasets_for_language(\"en\")}')
"

echo ""
echo "开始LSTM-CRF英文NER训练..."

# 1. 英文conll2003数据集 - LSTM-CRF模型
echo ""
echo "1. 训练英文conll2003数据集 (LSTM-CRF模型)"
echo "数据集: conll2003 (英文)"
echo "模型: LSTM-CRF"
echo "分词: 空格分词"

PYTHONPATH=. python -W ignore::FutureWarning scripts/train.py \
  --model_type lstm-crf \
  --dataset conll2003 \
  --lang en \
  --num_epochs 30 \
  --batch_size 256 \
  --max_len 128 \
  --learning_rate 0.001 \
  --weight_decay 0.0 \
  --embedding_dim 100 \
  --hidden_dim 256 \
  --log_dir ./logs/lstm_en \
  --processed_data_dir ./processed_data

# 2. 英文wikiann数据集 - LSTM-CRF模型
echo ""
echo "2. 训练英文wikiann数据集 (LSTM-CRF模型)"
echo "数据集: wikiann (英文)"
echo "模型: LSTM-CRF"
echo "分词: 空格分词"

PYTHONPATH=. python -W ignore::FutureWarning scripts/train.py \
  --model_type lstm-crf \
  --dataset wikiann \
  --lang en \
  --num_epochs 30 \
  --batch_size 256 \
  --max_len 128 \
  --learning_rate 0.001 \
  --weight_decay 0.0 \
  --embedding_dim 100 \
  --hidden_dim 256 \
  --log_dir ./logs/lstm_en \
  --processed_data_dir ./processed_data

# 3. 英文conll2012_ontonotesv5数据集 - LSTM-CRF模型
echo ""
echo "3. 训练英文conll2012_ontonotesv5数据集 (LSTM-CRF模型)"
echo "数据集: conll2012_ontonotesv5 (英文)"
echo "模型: LSTM-CRF"
echo "分词: 空格分词"
echo "注意: conll2012_ontonotesv5数据集不需要语言参数，但需要指定lang用于分词"

PYTHONPATH=. python -W ignore::FutureWarning scripts/train.py \
  --model_type lstm-crf \
  --dataset conll2012_ontonotesv5 \
  --lang en \
  --num_epochs 30 \
  --batch_size 256 \
  --max_len 128 \
  --learning_rate 0.001 \
  --weight_decay 0.0 \
  --embedding_dim 100 \
  --hidden_dim 256 \
  --log_dir ./logs/lstm_en \
  --processed_data_dir ./processed_data

echo ""
echo "LSTM-CRF英文NER训练完成！"
echo "日志文件保存在 ./logs/lstm_en 目录"
echo "处理后的数据缓存保存在 ./processed_data 目录"
echo ""
echo "训练结果文件："
echo "- LSTM-CRF + conll2003 (英文): ./logs/lstm_en/lstm-crf_conll2003_bert-base-cased.log"
echo "- LSTM-CRF + wikiann (英文): ./logs/lstm_en/lstm-crf_wikiann_en_bert-base-cased.log"
echo "- LSTM-CRF + conll2012_ontonotesv5 (英文): ./logs/lstm_en/lstm-crf_conll2012_ontonotesv5_bert-base-cased.log" 