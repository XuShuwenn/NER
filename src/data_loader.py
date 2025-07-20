import os
import pickle
from collections import Counter
from typing import List, Dict, Tuple, cast

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
from datasets import IterableDataset, IterableDatasetDict
from transformers import AutoTokenizer
import jieba

from src.utils.lang_utils import (
    get_language_config, 
    validate_language_dataset_combo,
    get_dataset_language_config,
    get_bert_model_for_language
)


# 分词器
def tokenize(sentence: str, lang: str) -> List[str]:
    """Tokenize a sentence based on language."""
    if lang == "zh":#中文分词
        return list(jieba.cut(sentence))
    else:
        return sentence.split()

# 构建词汇表
def build_vocab(sentences: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    """Build vocabulary with minimum frequency filtering."""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    word_counts = Counter()
    # 统计单词出现次数
    for sentence in sentences:
        for word in sentence:
            word_counts[word] += 1
    # 过滤低频词，以min_freq为阈值，高于min_freq的词才会被加入词汇表
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# 编码句子
def encode_sentences(sentences: List[List[str]], vocab: Dict[str, int], max_len: int) -> List[List[int]]:
    """Encode sentences to indices, pad/truncate to max_len."""
    encoded = []
    # 遍历句子列表
    for sent in sentences:
        # 将句子中的每个单词转换为词汇表中的索引
        ids = [vocab.get(tok, vocab["<UNK>"]) for tok in sent]
        # 如果句子长度超过max_len，则截断句子
        ids = ids[:max_len] + [vocab["<PAD>"]] * (max_len - len(ids))
        # 将句子转换为索引列表
        encoded.append(ids)
    return encoded

# 编码标签
def encode_labels(labels: List[List[int]], pad_id: int, max_len: int) -> List[List[int]]:
    """Pad/truncate label sequences to max_len using pad_id."""
    encoded = []
    # 遍历标签列表
    for label_seq in labels:
        # 如果标签序列长度超过max_len，则截断标签序列
        ids = label_seq[:max_len] + [pad_id] * (max_len - len(label_seq))
        encoded.append(ids)
    return encoded

# 加载预训练词嵌入
def load_pretrained_embeddings(vocab: Dict[str, int], embedding_path: str, embedding_dim: int) -> np.ndarray:
    """Load pretrained embeddings for vocab."""
    # 随机初始化嵌入矩阵，大小为词汇表大小乘以嵌入维度
    embedding_matrix = np.random.normal(size=(len(vocab), embedding_dim)).astype(np.float32)
    found = 0
    # 加载预训练词嵌入
    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in f:
            # 分割行
            values = line.rstrip().split()
            # 获取单词
            word = values[0]
            # 如果单词在词汇表中，则更新嵌入矩阵
            if word in vocab:
                embedding_matrix[vocab[word]] = np.array(values[1:], dtype=np.float32)
                found += 1
    # 打印加载的预训练词嵌入数量
    print(f"Loaded {found} pretrained embeddings.")
    return embedding_matrix

# NER数据集类
class NERDataset(Dataset):
    """Dataset for NER"""
    def __init__(self, input_ids, label_ids, mask):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.mask = mask

    # 获取数据集的第idx个样本
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.label_ids[idx], dtype=torch.long),
            "mask": torch.tensor(self.mask[idx], dtype=torch.bool)
        }

    # 获取数据集的长度
    def __len__(self):
        return len(self.input_ids)

# 加载和预处理数据
def load_and_preprocess_data(
    dataset_name: str,
    lang: str,
    processed_data_dir: str,
    max_len: int = 128,
    min_freq: int = 1,
    force_preprocess: bool = False,
    model_type: str = "lstm",
    model_name_or_path: str = "bert-base-cased"
) -> Tuple[Dict[str, NERDataset], List[str], Dict[int, str], Dict[str, int], Dict[str, int] | None]:
    """
    加载和预处理NER数据,缓存处理后的数据。
    返回值：
    字典(split: NERDataset),标签列表,id2label,label2id,词汇表(None for BERT),用于训练和评估的NER数据集
    """
    # 创建处理后的数据目录
    os.makedirs(processed_data_dir, exist_ok=True)
    # 创建缓存路径
    cache_path = os.path.join(processed_data_dir, f"{dataset_name}_{lang}_{model_type}_cache.pkl")
    # 如果缓存文件存在且不强制重新处理，则加载缓存数据
    if os.path.exists(cache_path) and not force_preprocess:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 验证语言和数据集组合
    if not validate_language_dataset_combo(lang, dataset_name):
        raise ValueError(f"Dataset {dataset_name} is not supported for language {lang}")
    
    # 获取数据集配置
    dataset_config = get_dataset_language_config(dataset_name, lang)
    
    # 加载原始数据集
    if dataset_name == "conll2003":
        dataset = load_dataset(dataset_name)
    elif dataset_name == "wikiann":
        # 获取WikiAnn数据集的语言配置
        lang_config = get_language_config(lang)
        wikiann_config = lang_config.get("wikiann_config", lang)
        dataset = load_dataset(dataset_name, wikiann_config)
    elif dataset_name == "conll2012_ontonotesv5":
        # conll2012_ontonotesv5 需要指定配置名称
        lang_config = get_language_config(lang)
        config_name = lang_config.get("conll2012_config", "english_v4")
        print(f"Loading conll2012_ontonotesv5 with config: {config_name}")
        dataset = load_dataset(dataset_name, config_name)
    else:
        dataset = load_dataset(dataset_name, lang)  # lang must be a valid config, not None
    # 如果加载的数据集是IterableDatasetDict，则抛出错误
    if isinstance(dataset, IterableDatasetDict):
        raise ValueError("Loaded dataset is an IterableDatasetDict (streaming mode). Please load without streaming for indexable access.")
    # 获取数据集的分割
    splits = ["train", "validation", "test"]
    data = {}
    # 遍历数据集的分割
    for split in splits:
        try:
            split_data = dataset[split]
            # 如果分割数据是IterableDataset，则跳过
            if isinstance(split_data, IterableDataset):
                continue
            # 如果分割数据有__getitem__方法，则添加到数据字典中
            if hasattr(split_data, "__getitem__"):
                data[split] = split_data
        except Exception:
            continue
    
    # 处理不同数据集结构
    if dataset_name == "conll2012_ontonotesv5":
        # 特殊处理conll2012_ontonotesv5嵌套结构
        label_names = None
        label_field = None
        
        # 在嵌套结构中找到标签字段
        for split in data:
            if len(data[split]) > 0:
                # 获取数据集特征以找到named_entities
                dataset_features = data[split].features
                if "sentences" in dataset_features:
                    sentence_features = dataset_features["sentences"][0]
                    for candidate in ["named_entities", "ner_tags", "labels", "ner"]:
                        if candidate in sentence_features:
                            label_names = sentence_features[candidate].feature.names
                            label_field = candidate
                            break
                    if label_names is not None:
                        break
        
        if label_names is None:
            raise ValueError("Could not find label names in conll2012_ontonotesv5 dataset features.")
    else:
        # 标准处理其他数据集
        label_names = None
        label_field = None
        for split in data:
            for candidate in ["ner_tags", "labels", "ner"]:
                if candidate in data[split].features:
                    label_names = data[split].features[candidate].feature.names
                    label_field = candidate
                    break
            if label_names is not None:
                break
        if label_names is None:
            raise ValueError("Could not find label names in dataset features.")

    label2id = {l: i for i, l in enumerate(label_names)}
    id2label = {i: l for l, i in label2id.items()}
    pad_label_id = label2id["O"] if "O" in label2id else 0

    # 分词并收集句子/标签
    all_sentences = []
    all_labels = []
    
    for split in data:
        split_data = data[split]
        if isinstance(split_data, IterableDataset):
            continue
        split_data = cast(HFDataset, split_data)
        for i in range(len(split_data)):
            x = split_data[i]
            
            if dataset_name == "conll2012_ontonotesv5":
                # 处理conll2012_ontonotesv5嵌套结构
                if "sentences" in x:
                    for sentence in x["sentences"]:
                        if "words" in sentence and label_field in sentence:
                            words = sentence["words"]
                            labels = sentence[label_field]
                            if len(words) == len(labels):
                                all_sentences.append(words)
                                all_labels.append(labels)
            else:
                # 标准处理其他数据集
                # 如果tokens可用，则直接使用tokens
                if "tokens" in x:
                    sents = x["tokens"]
                elif "text" in x:
                    sents = tokenize(x["text"], lang)
                else:
                    raise ValueError("No tokens or text found in data sample.")
                all_sentences.append(sents)
                # 直接使用标签索引
                if label_field in x:
                    labels = x[label_field]
                else:
                    raise ValueError(f"No {label_field} found in data sample.")
                all_labels.append(labels)

    # 处理BERT和LSTM不同
    if model_type == "bert":
        # 对于BERT，根据语言选择合适的模型
        if model_name_or_path == "bert-base-cased" and lang != "en":
            # 如果使用默认英文模型但语言不是英文，自动选择合适的模型
            model_name_or_path = get_bert_model_for_language(lang)
        print(f"Using tokenizer for model: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        vocab = None  # BERT uses its own vocabulary
    else:
        # 对于LSTM，构建自定义词汇表
        vocab = build_vocab(all_sentences, min_freq)

    # 编码和填充
    processed = {}
    for split in data:
        split_data = data[split]
        if isinstance(split_data, IterableDataset):
            continue
        split_data = cast(HFDataset, split_data)
        split_sentences = []
        split_labels = []
        
        for i in range(len(split_data)):
            x = split_data[i]
            
            if dataset_name == "conll2012_ontonotesv5":
                # 处理conll2012_ontonotesv5嵌套结构
                if "sentences" in x:
                    for sentence in x["sentences"]:
                        if "words" in sentence and label_field in sentence:
                            words = sentence["words"]
                            labels = sentence[label_field]
                            if len(words) == len(labels):
                                split_sentences.append(words)
                                split_labels.append(labels)
            else:
                # 标准处理其他数据集
                if "tokens" in x:
                    sents = x["tokens"]
                elif "text" in x:
                    sents = tokenize(x["text"], lang)
                else:
                    raise ValueError("No tokens or text found in data sample.")
                split_sentences.append(sents)
                if label_field in x:
                    labels = x[label_field]
                else:
                    raise ValueError(f"No {label_field} found in data sample.")
                split_labels.append(labels)
        
        if model_type == "bert":
            # 使用BERT分词器进行编码
            input_ids = []
            attention_masks = []
            label_ids = []
            
            for sentence, labels in zip(split_sentences, split_labels):
                # 使用空格连接单词以用于BERT分词器
                text = " ".join(sentence)
                tokenized = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors=None  # 返回Python列表而不是张量
                )
                
                # 对齐标签与BERT标记
                word_ids = tokenized.word_ids()
                aligned_labels = []
                prev_word_id = None
                
                for word_id in word_ids:
                    if word_id is None:
                        # 特殊标记 (CLS, SEP, PAD)
                        aligned_labels.append(-100)
                    elif word_id != prev_word_id:
                        # 新单词
                        if word_id < len(labels):
                            aligned_labels.append(labels[word_id])
                        else:
                            aligned_labels.append(-100)  # Out of bounds
                        prev_word_id = word_id
                    else:
                        # Subword token, use -100 (ignore in loss)
                        aligned_labels.append(-100)
                
                input_ids.append(tokenized["input_ids"])
                attention_masks.append(tokenized["attention_mask"])
                label_ids.append(aligned_labels)
            
            # Create mask from attention_mask
            mask = attention_masks
        else:
            # Use custom vocabulary for LSTM
            if vocab is None:
                raise ValueError("Vocabulary is required for LSTM models")
            input_ids = encode_sentences(split_sentences, vocab, max_len)
            label_ids = encode_labels(split_labels, pad_label_id, max_len)
            mask = [[int(tok != vocab["<PAD>"]) for tok in seq] for seq in input_ids]
        
        processed[split] = NERDataset(input_ids, label_ids, mask)
    # 缓存处理后的数据
    result = (processed, label_names, id2label, label2id, vocab)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result
