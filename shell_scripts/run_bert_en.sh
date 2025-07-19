#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_MODULES_CACHE=$HOME/.cache/huggingface/modules

echo "============================================================"
echo "BERT英文NER训练脚本"
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
echo "开始BERT英文NER训练..."

# 1. 英文conll2003数据集 - BERT模型
echo ""
echo "1. 训练英文conll2003数据集 (BERT模型)"
echo "数据集: conll2003 (英文)"
echo "模型: BERT (bert-base-cased)"
echo "分词: BERT英文分词器"

PYTHONPATH=. python -W ignore::FutureWarning scripts/train.py \
  --model_type bert \
  --dataset conll2003 \
  --lang en \
  --model_name_or_path bert-base-cased \
  --num_epochs 10 \
  --batch_size 128 \
  --max_len 128 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --log_dir ./logs/bert_en \
  --processed_data_dir ./processed_data

# 2. 英文wikiann数据集 - BERT模型
echo ""
echo "2. 训练英文wikiann数据集 (BERT模型)"
echo "数据集: wikiann (英文)"
echo "模型: BERT (bert-base-cased)"
echo "分词: BERT英文分词器"

PYTHONPATH=. python -W ignore::FutureWarning scripts/train.py \
  --model_type bert \
  --dataset wikiann \
  --lang en \
  --model_name_or_path bert-base-cased \
  --num_epochs 10 \
  --batch_size 128 \
  --max_len 128 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --log_dir ./logs/bert_en \
  --processed_data_dir ./processed_data

# 3. 英文conll2012_ontonotesv5数据集 - BERT模型
echo ""
echo "3. 训练英文conll2012_ontonotesv5数据集 (BERT模型)"
echo "数据集: conll2012_ontonotesv5 (英文)"
echo "模型: BERT (bert-base-cased)"
echo "分词: BERT英文分词器"
echo "注意: conll2012_ontonotesv5数据集不需要语言参数，但需要指定lang用于分词和模型选择"

PYTHONPATH=. python -W ignore::FutureWarning scripts/train.py \
  --model_type bert \
  --dataset conll2012_ontonotesv5 \
  --lang en \
  --model_name_or_path bert-base-cased \
  --num_epochs 10 \
  --batch_size 128 \
  --max_len 128 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --log_dir ./logs/bert_en \
  --processed_data_dir ./processed_data

echo ""
echo "BERT英文NER训练完成！"
echo "日志文件保存在 ./logs/bert_en 目录"
echo "处理后的数据缓存保存在 ./processed_data 目录"
echo ""
echo "训练结果文件："
echo "- BERT + conll2003 (英文): ./logs/bert_en/bert_conll2003_bert-base-cased.log"
echo "- BERT + wikiann (英文): ./logs/bert_en/bert_wikiann_en_bert-base-cased.log"
echo "- BERT + conll2012_ontonotesv5 (英文): ./logs/bert_en/bert_conll2012_ontonotesv5_bert-base-cased.log" 