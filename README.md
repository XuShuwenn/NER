# NER (命名实体识别)项目

本项目支持基于BERT和LSTM-CRF的多语言命名实体识别（NER）任务，支持常见数据集（如conll2003、conll2012_ontonotesv5、wikiann），并支持多GPU并行训练、日志自动管理和大文件自动忽略。

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
│   ├── plot_logs.py
│   └── preprocess.py
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
├── data/                
├── processed_data/      
├── logs/               
├── wandb/               
├── .conda/              
├── .vscode/            
└── .git/                
```

---

## 支持的数据集
- `conll2003`
- `conll2012_ontonotesv5`
- `wikiann`

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
bash shell_scripts/run_ner.sh
```
- 自动遍历所有模型和数据集组合，支持多GPU（已在脚本开头设置`CUDA_VISIBLE_DEVICES`）。
- 日志保存在`logs/`目录，已自动忽略。

### 2. 单模型单数据集训练

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
- 其它参数可参考`scripts/arg_utils.py`。

---

## 日志与大文件管理
- `logs/`、`processed_data/`、`data/`、`wandb/`、模型权重等大文件已在`.gitignore`中自动忽略。
- 不会被git提交，避免GitHub大文件报错。
- 日志文件自动按模型+数据集命名，保存在`logs/`目录。

---

## 多GPU训练
- 已自动支持`torch.nn.DataParallel`，脚本会自动利用所有可见GPU。
- 在`shell_scripts/run_ner.sh`开头设置`export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7,8`，（可根据实际GPU调整）。
- 训练时会自动打印`Using X GPUs for DataParallel!`

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


