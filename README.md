# NER (命名实体识别)项目

本项目支持基于BERT和LSTM-CRF的多语言命名实体识别（NER）任务，支持多种语言（中文、英文、德文、西班牙文等）和常见数据集（如conll2003、conll2012_ontonotesv5、wikiann），并支持单GPU训练、日志自动管理和大文件自动忽略。

---

## 目录结构

```
NER/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── train.py
│   └── plot_logs.py
├── shell_scripts/
│   └── run_ner.sh
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bert_ner.py
│   │   └── lstm_ner.py
│   └── utils/
│       ├── __init__.py
│       ├── arg_utils.py
│       └── logging_utils.py
├── processed_data/      
├── logs/               
│   ├── bert_en/        
│   ├── bert_zh/        
│   ├── lstm_en/        
│   └── lstm_zh/        
├── log_figures/       
│   ├── bert_en/        
│   ├── bert_zh/        
│   ├── lstm_en/        
│   └── lstm_zh/        
├── wandb/               
├── .conda/              
├── .vscode/            
└── .git/                
```

---

## 支持的语言和数据集

### 支持的语言
- **中文 (zh)**: 使用jieba分词，支持BERT模型
- **英文 (en)**: 空格分词，支持BERT模型

### 支持的数据集
- `conll2003` (英文) - 不需要语言参数
- `conll2012_ontonotesv5` (多语言) - 不需要语言参数，但需要指定lang用于分词和模型选择
- `wikiann` (多语言) - 需要指定语言配置

### 语言和数据集组合
查看所有支持的语言和数据集组合：
```bash
python scripts/train.py --show_languages
```

## 支持的模型
- `bert`（支持多语言BERT、bert-base-cased等）
- `lstm-crf`（可选预训练词向量）

---

## 环境依赖

建议使用conda环境，安装依赖：

```bash
conda create -n ner python=3.10
conda activate ner
pip install -r requirements.txt
```

---

## 训练用法

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
- 自动遍历所有模型和数据集组合，使用单GPU训练。
- 日志按类别分别保存在`logs/bert_en/`、`logs/bert_zh/`、`logs/lstm_en/`、`logs/lstm_zh/`目录。

### 2. 单模型单数据集训练

#### 英文训练示例
```bash
python scripts/train.py \
  --model_type bert \
  --dataset conll2003 \
  --lang en \
  --model_name_or_path bert-base-cased \
  --num_epochs 20 \
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

- 其它参数可参考`scripts/arg_utils.py`。

---

## 日志与大文件管理
- `logs/`、`processed_data/`、`wandb/`、模型权重等大文件已在`.gitignore`中自动忽略。
- 不会被git提交，避免GitHub大文件报错。
- 日志文件自动按模型+数据集命名，保存在`logs/`目录。

---

## GPU训练
- 使用单GPU进行训练，默认使用GPU 0。
- 在shell脚本开头设置`export CUDA_VISIBLE_DEVICES=0`。
- 支持CUDA和CPU训练，自动检测可用设备。

---

## 可视化
- 训练日志可用`scripts/plot_logs.py`批量可视化loss、F1等曲线。
- 运行：
  ```bash
  python scripts/plot_logs.py
  ```
  图片保存在`log_figures/`目录。

---

## 常见问题

### 1. push时报大文件错误？
- 只需`git rm --cached <大文件>`，并确保`.gitignore`已包含相关目录和文件类型。
- 重新commit并push即可。

### 2. acc收敛但F1持续提升？
- NER任务以F1为主，acc仅供参考。
- 只要F1持续提升，模型就是在进步。

### 3. 日志/数据/模型不会被git追踪？
- `.gitignore`已自动忽略所有大文件和缓存。


