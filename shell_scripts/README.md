# 训练脚本说明

本项目提供了4个专门的训练脚本，分别用于不同模型和语言的NER训练。

## 📁 脚本文件

| 脚本名称 | 模型类型 | 语言 | 数据集 | 用途 |
|---------|---------|------|--------|------|
| `run_bert_en.sh` | BERT | 英文 | conll2003, wikiann, conll2012_ontonotesv5 | BERT英文训练 |
| `run_bert_zh.sh` | BERT | 中文 | wikiann, conll2012_ontonotesv5 | BERT中文训练 |
| `run_lstm_en.sh` | LSTM-CRF | 英文 | conll2003, wikiann, conll2012_ontonotesv5 | LSTM英文训练 |
| `run_lstm_zh.sh` | LSTM-CRF | 中文 | wikiann, conll2012_ontonotesv5 | LSTM中文训练 |

## 🚀 使用方法

### 1. BERT英文训练
```bash
bash shell_scripts/run_bert_en.sh
```
**训练内容：**
- conll2003 (英文)
- wikiann (英文)
- conll2012_ontonotesv5 (英文)

**参数配置：**
- 学习率: 4e-5
- 批量大小: 128
- 训练轮数: 10
- 权重衰减: 0.01

### 2. BERT中文训练
```bash
bash shell_scripts/run_bert_zh.sh
```
**训练内容：**
- wikiann (中文)
- conll2012_ontonotesv5 (中文)

**参数配置：**
- 学习率: 5e-5
- 批量大小: 32
- 训练轮数: 10
- 权重衰减: 0.01

### 3. LSTM英文训练
```bash
bash shell_scripts/run_lstm_en.sh
```
**训练内容：**
- conll2003 (英文)
- wikiann (英文)
- conll2012_ontonotesv5 (英文)

**参数配置：**
- 学习率: 0.001
- 批量大小: 256
- 训练轮数: 30
- 嵌入维度: 100
- 隐藏维度: 256

### 4. LSTM中文训练
```bash
bash shell_scripts/run_lstm_zh.sh
```
**训练内容：**
- wikiann (中文)
- conll2012_ontonotesv5 (中文)

**参数配置：**
- 学习率: 0.001
- 批量大小: 256
- 训练轮数: 30
- 嵌入维度: 100
- 隐藏维度: 256

## 📊 日志文件命名规则

### BERT模型
- 英文conll2003: `bert_conll2003_bert-base-cased.log`
- 英文wikiann: `bert_wikiann_en_bert-base-cased.log`
- 英文conll2012: `bert_conll2012_ontonotesv5_bert-base-cased.log`
- 中文wikiann: `bert_wikiann_zh_bert-base-cased.log`
- 中文conll2012: `bert_conll2012_ontonotesv5_bert-base-cased.log`

### LSTM-CRF模型
- 英文conll2003: `lstm-crf_conll2003_bert-base-cased.log`
- 英文wikiann: `lstm-crf_wikiann_en_bert-base-cased.log`
- 英文conll2012: `lstm-crf_conll2012_ontonotesv5_bert-base-cased.log`
- 中文wikiann: `lstm-crf_wikiann_zh_bert-base-cased.log`
- 中文conll2012: `lstm-crf_conll2012_ontonotesv5_bert-base-cased.log`

## 🔧 环境配置

所有脚本都包含以下环境配置：
```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_MODULES_CACHE=$HOME/.cache/huggingface/modules
```

## 📝 注意事项

1. **数据集限制**: 中文不支持conll2003数据集
2. **分词方式**: 
   - 英文使用空格分词
   - 中文使用jieba分词
3. **模型差异**:
   - BERT使用预训练模型，训练轮数较少
   - LSTM-CRF从头训练，训练轮数较多
4. **批量大小**: BERT使用较小批量，LSTM使用较大批量
5. **日志文件**: 所有日志文件都保存在`./logs`目录

## 🎯 推荐使用场景

- **快速实验**: 使用BERT脚本，训练速度快
- **深入研究**: 使用LSTM脚本，可以更好地理解模型行为
- **英文任务**: 优先使用英文脚本，数据集更丰富
- **中文任务**: 使用中文脚本，专门针对中文优化

## 🔍 监控训练

训练过程中可以查看日志文件：
```bash
# 实时查看训练日志
tail -f logs/bert_wikiann_en_bert-base-cased.log

# 查看所有日志文件
ls -la logs/
```