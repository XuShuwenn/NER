# 命名实体识别 (NER) 项目 (v2)

本项目是一个基于PyTorch和Hugging Face生态的命名实体识别项目。项目内置了对BERT和BiLSTM-CRF模型的支持，并使用 `datasets` 库加载数据集，通过 `wandb` 进行实验跟踪和可视化。

## 主要技术栈
- **核心框架**: PyTorch
- **数据集管理**: Hugging Face `datasets`
- **模型与分词**: Hugging Face `transformers`
- **实验跟踪**: Weights & Biases (`wandb`)

## 项目框架

由于使用 `datasets` 库进行数据管理，我们不再需要本地的 `data` 目录来存放原始和处理后数据，项目结构更加简洁。

```
NER/
├── README.md                # 项目介绍
├── scripts/                 # 脚本目录
│   ├── preprocess.py        # 数据预处理/特征工程脚本
│   ├── train.py             # 模型训练脚本
│   └── evaluate.py          # 模型评估脚本
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载和处理模块
│   ├── models/              # 模型定义目录
│   │   ├── __init__.py
│   │   ├── bert_ner.py      # BERT-NER 模型
│   │   └── lstm_crf_ner.py  # BiLSTM-CRF 模型
│   └── utils.py             # 工具函数
├── config.py                  # 配置文件 (模型参数、路径、W&B项目名)
├── requirements.txt         # Python 依赖
└── .gitignore               # Git忽略文件
```

## 环境准备

1.  **创建 Conda 环境**
    ```bash
    conda create -n ner_env python=3.8
    conda activate ner_env
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

3.  **登录 WandB**
    在终端运行 `wandb login` 并根据提示输入你的 API Key。
    ```bash
    wandb login
    ```

## 数据集

本项目使用 Hugging Face `datasets` 库动态加载以下数据集，无需手动下载：
1.  **conll2003**: 经典的英文NER基准数据集。
2.  **wikiann**: 多语言NER数据集，我们将使用其中的中文部分 (`zh`)。
3.  **zju-chinai/ontonotes_cn**: 中文版的 OntoNotes 数据集。

数据加载和预处理逻辑位于 `src/data_loader.py` 中。

## 模型训练

通过 `scripts/train.py` 脚本来训练模型。所有实验配置（如模型、数据集、超参数）都在 `config.py` 文件中定义。训练过程中的所有指标（如loss, F1-score）将会自动上传到 `wandb`。

```bash
# 训练 BERT 模型在 conll2003 数据集上
python scripts/train.py --model_name bert --dataset conll2003

# 训练 BiLSTM-CRF 模型在 wikiann (中文) 数据集上
python scripts/train.py --model_name lstm --dataset wikiann-zh
```

## 模型评估

使用 `scripts/evaluate.py` 来评估已训练模型的性能。

```bash
python scripts/evaluate.py --model_path path/to/your/trained/model --dataset conll2003
```
