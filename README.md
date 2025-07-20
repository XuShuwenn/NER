# NER (命名实体识别)

本项目支持基于BERT和LSTM-CRF的多语言命名实体识别（NER）任务，支持中文和英文两种语言，支持多种数据集（CoNLL-2003、WikiAnn、CoNLL-2012 OntoNotes v5），并支持单GPU训练、日志自动管理、可视化分析和实验报告生成。

---

## 📁 目录结构

```
NER/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── train.py              # 主训练脚本
│   └── plot_logs.py          # 日志可视化脚本
├── shell_scripts/
│   ├── README.md             # 训练脚本说明
│   ├── run_bert_en.sh        # BERT英文训练脚本
│   ├── run_bert_zh.sh        # BERT中文训练脚本
│   ├── run_lstm_en.sh        # LSTM英文训练脚本
│   └── run_lstm_zh.sh        # LSTM中文训练脚本
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # 数据加载和预处理
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bert_ner.py       # BERT模型定义
│   │   └── lstm_ner.py       # LSTM-CRF模型定义
│   └── utils/
│       ├── __init__.py
│       ├── arg_utils.py      # 参数处理工具
│       ├── lang_utils.py     # 语言配置工具
│       └── logging_utils.py  # 日志工具
├── processed_data/           # 预处理后的数据缓存
├── logs/                     # 训练日志
│   ├── bert_en/             # BERT英文训练日志
│   ├── bert_zh/             # BERT中文训练日志
│   ├── lstm_en/             # LSTM英文训练日志
│   └── lstm_zh/             # LSTM中文训练日志
├── log_figures/             # 训练曲线图
│   ├── bert_en/             # BERT英文训练图表
│   ├── bert_zh/             # BERT中文训练图表
│   ├── lstm_en/             # LSTM英文训练图表
│   └── lstm_zh/             # LSTM中文训练图表
├── report/                  # 实验报告
│   └── report.tex           # LaTeX格式的详细实验报告
├── wandb/                   # Weights & Biases日志
├── .conda/                  # Conda环境
├── .vscode/                 # VSCode配置
└── .git/                    # Git版本控制
```

---

## 🎯 项目特色

### 支持的语言和数据集
- **中文 (zh)**: 使用jieba分词，支持BERT和LSTM-CRF模型
- **英文 (en)**: 使用空格分词，支持BERT和LSTM-CRF模型

### 支持的数据集
- `conll2003` (英文) - 经典的NER数据集
- `wikiann` (多语言) - 维基百科标注的NER数据集
- `conll2012_ontonotesv5` (多语言) - OntoNotes项目的大型NER数据集

### 支持的模型
- **BERT**: 基于Transformer的预训练语言模型，支持多语言
- **LSTM-CRF**: 结合双向LSTM和CRF层的序列标注模型

---

## 📊 实验结果

基于实际实验数据，主要发现：

### 英文数据集表现
- **CoNLL-2003**: BERT模型F1分数86.01%，显著优于LSTM-CRF的73.00%
- **WikiAnn英文**: BERT模型F1分数76.63%，优于LSTM-CRF的61.23%
- **CoNLL-2012英文**: BERT模型F1分数82.18%，优于LSTM-CRF的73.94%

### 中文数据集表现
- **WikiAnn中文**: BERT模型F1分数80.47%，优于LSTM-CRF的68.66%
- **CoNLL-2012中文**: 意外发现LSTM-CRF模型F1分数60.38%，优于BERT的52.24%

### 关键发现
1. BERT模型在英文任务上表现优异，F1分数提升8-15个百分点
2. 中文NER任务仍面临挑战，需要更好的分词策略和预训练模型
3. LSTM-CRF在特定中文数据集上可能表现更好，具有计算效率优势

---

## 🚀 环境依赖

建议使用conda环境，安装依赖：

```bash
conda create -n ner python=3.10
conda activate ner
pip install -r requirements.txt
```

### 主要依赖
- **PyTorch**: >=1.10.0
- **Transformers**: 4.35.2
- **Datasets**: 2.15.0
- **pytorch-crf**: 0.7.2
- **seqeval**: 1.2.2
- **jieba**: >=0.42.1 (中文分词)
- **matplotlib**: >=3.5.0 (可视化)

---

## 🎮 使用方法

### 1. 批量训练（推荐）

```bash
# BERT英文训练
bash shell_scripts/run_bert_en.sh

# BERT中文训练
bash shell_scripts/run_bert_zh.sh

# LSTM英文训练
bash shell_scripts/run_lstm_en.sh

# LSTM中文训练
bash shell_scripts/run_lstm_zh.sh
```

**特点：**
- 自动遍历所有模型和数据集组合
- 使用单GPU训练，自动设置环境变量
- 日志按类别分别保存在对应目录
- 支持CUDA和CPU训练，自动检测可用设备

### 2. 单模型单数据集训练

#### 英文训练示例
```bash
python scripts/train.py \
  --model_type bert \
  --dataset conll2003 \
  --lang en \
  --model_name_or_path bert-base-cased \
  --num_epochs 10 \
  --batch_size 128 \
  --max_len 128 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --log_dir ./logs \
  --processed_data_dir ./processed_data
```

#### 中文训练示例
```bash
python scripts/train.py \
  --model_type lstm-crf \
  --dataset wikiann \
  --lang zh \
  --num_epochs 30 \
  --batch_size 256 \
  --max_len 128 \
  --learning_rate 0.001 \
  --embedding_dim 100 \
  --hidden_dim 256 \
  --log_dir ./logs \
  --processed_data_dir ./processed_data
```

### 3. 查看支持的语言和数据集
```bash
python scripts/train.py --show_languages
```

---

## 📈 可视化分析

### 训练曲线可视化
```bash
python scripts/plot_logs.py
```

**功能特点：**
- 自动递归查找所有日志文件
- 生成训练损失、验证损失、F1分数、准确率曲线
- 图片保存在`log_figures/`目录，保持原有目录结构
- 支持批量处理多个实验日志

### 实验报告
项目包含详细的LaTeX格式实验报告：
- 位置：`report/report.tex`
- 内容：基于实际实验数据的详细分析
- 包括：实验结果对比、训练过程分析、局限性讨论、未来工作建议

---

## 🔧 技术特性

### 数据处理
- **智能缓存**: 预处理数据自动缓存，避免重复处理
- **多语言支持**: 中文使用jieba分词，英文使用空格分词
- **数据验证**: 自动验证语言和数据集组合的有效性
- **类别权重**: BERT模型使用类别权重处理数据不平衡

### 模型架构
- **BERT模型**: 基于预训练Transformer，支持多语言
- **LSTM-CRF模型**: 双向LSTM + CRF层，支持预训练词向量
- **灵活配置**: 支持自定义超参数和模型配置

---

## 📝 日志管理

### 日志文件命名规则
- **BERT模型**: `bert_{dataset}_{lang}_{model_name}.log`
- **LSTM模型**: `lstm-crf_{dataset}_{lang}_{model_name}.log`

### 日志内容
- 训练损失和验证损失
- 准确率和F1分数
- 精确率和召回率

### 大文件管理
- `logs/`、`processed_data/`、`wandb/`等大文件已在`.gitignore`中自动忽略
- 不会被git提交，避免GitHub大文件报错
- 支持实验数据的本地缓存和清理

---

## 🐛 常见问题

### 1. CUDA内存不足？
- 减少batch_size（建议BERT: 32-128, LSTM: 128-256）
- 减少max_len（建议128-256）
- 使用梯度累积模拟大批次训练

### 2. 中文模型性能较差？
- 尝试不同的中文预训练模型
- 调整学习率和训练轮数
- 检查分词质量和数据预处理

### 3. 训练收敛但F1分数低？
- NER任务以F1为主，准确率仅供参考
- 检查数据标注质量和类别分布
- 尝试调整类别权重和损失函数

### 4. 日志文件过大？
- 定期清理旧的日志文件
- 使用`plot_logs.py`生成图表后删除原始日志
- 配置日志轮转和压缩





