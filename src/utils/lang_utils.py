import jieba
import re
from typing import List, Dict, Optional
from transformers import AutoTokenizer


# 支持的语言配置（仅中文和英文）
SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "tokenizer": "space",  # 空格分词
        "bert_model": "bert-base-cased",
        "datasets": ["conll2003", "wikiann", "conll2012_ontonotesv5"],
        "wikiann_config": "english_v4",  # WikiAnn数据集配置
        "conll2012_config": "english_v4"  # conll2012_ontonotesv5数据集配置
    },
    "zh": {
        "name": "Chinese",
        "tokenizer": "jieba",  # jieba分词
        "bert_model": "bert-base-chinese",  # 使用专门的中文BERT模型
        "datasets": ["wikiann", "conll2012_ontonotesv5"],
        "wikiann_config": "zh",  # WikiAnn数据集配置
        "conll2012_config": "chinese_v4"  # conll2012_ontonotesv5数据集配置
    }
}


def get_language_config(lang: str) -> Dict:
    """
    获取语言配置
    Args:
        lang: 语言代码 (en, zh)
    Returns:
        语言配置字典
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {lang}. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
    
    return SUPPORTED_LANGUAGES[lang]


def get_bert_model_for_language(lang: str) -> str:
    """
    根据语言获取合适的BERT模型
    Args:
        lang: 语言代码
    Returns:
        BERT模型名称
    """
    config = get_language_config(lang)
    return config["bert_model"]


def tokenize_text(text: str, lang: str) -> List[str]:
    """
    根据语言对文本进行分词
    Args:
        text: 输入文本
        lang: 语言代码
    Returns:
        分词结果列表
    """
    config = get_language_config(lang)
    
    if config["tokenizer"] == "jieba":
        # 中文分词
        return list(jieba.cut(text))
    elif config["tokenizer"] == "space":
        # 空格分词（适用于英文等）
        return text.split()
    else:
        raise ValueError(f"Unknown tokenizer type: {config['tokenizer']}")


def get_supported_datasets_for_language(lang: str) -> List[str]:
    """
    获取指定语言支持的数据集列表
    Args:
        lang: 语言代码
    Returns:
        支持的数据集列表
    """
    config = get_language_config(lang)
    return config["datasets"]


def validate_language_dataset_combo(lang: str, dataset: str) -> bool:
    """
    验证语言和数据集组合是否有效
    Args:
        lang: 语言代码
        dataset: 数据集名称
    Returns:
        是否有效
    """
    try:
        config = get_language_config(lang)
        return dataset in config["datasets"]
    except ValueError:
        return False


def get_dataset_language_config(dataset: str, lang: str) -> Dict:
    """
    获取数据集的语言特定配置
    Args:
        dataset: 数据集名称
        lang: 语言代码
    Returns:
        数据集配置
    """
    if not validate_language_dataset_combo(lang, dataset):
        raise ValueError(f"Dataset {dataset} is not supported for language {lang}")
    
    # 数据集特定的配置
    dataset_configs = {
        "wikiann": {
            "lang": lang,  # wikiann需要语言参数
            "text_field": "tokens",
            "label_field": "ner_tags"
        },
        "conll2003": {
            "lang": None,  # conll2003不需要语言参数
            "text_field": "tokens", 
            "label_field": "ner_tags"
        },
        "conll2012_ontonotesv5": {
            "lang": None,  # conll2012_ontonotesv5不需要语言参数
            "text_field": "words",
            "label_field": "named_entities",
            "config": "english_v4"  # 默认配置
        }
    }
    
    if dataset not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return dataset_configs[dataset]

def print_language_support_info():
    """
    打印支持的语言和数据集信息
    """
    print("=" * 60)
    print("支持的语言和数据集配置")
    print("=" * 60)
    
    for lang, config in SUPPORTED_LANGUAGES.items():
        print(f"\n语言: {lang} ({config['name']})")
        print(f"  分词器: {config['tokenizer']}")
        print(f"  BERT模型: {config['bert_model']}")
        print(f"  支持的数据集: {', '.join(config['datasets'])}")
    
    print("\n" + "=" * 60)
    print("使用示例:")
    print("  # 英文conll2003")
    print("  --dataset conll2003 --lang en")
    print("  # 中文wikiann") 
    print("  --dataset wikiann --lang zh")
    print("  # 英文wikiann")
    print("  --dataset wikiann --lang en")
    print("=" * 60)


if __name__ == "__main__":
    print_language_support_info() 