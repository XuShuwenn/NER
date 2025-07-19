import argparse
from src.utils.lang_utils import validate_language_dataset_combo, print_language_support_info

def get_train_args():
    """
    获取训练参数
    """
    parser = argparse.ArgumentParser(description="Train a NER model.")
    parser.add_argument("--model_type", type=str, required=True, choices=["bert", "lstm-crf"])
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., conll2003, wikiann, conll2012_ontonotesv5)")
    parser.add_argument("--lang", type=str, default="en", help="Language/config for the dataset (e.g., 'en', 'zh')")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased", help="Huggingface model name or path")
    parser.add_argument("--embedding_path", type=str, default=None, help="Path to pretrained embeddings for LSTM")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Embedding dimension for LSTM")
    parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM hidden dimension")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for log files")
    parser.add_argument("--processed_data_dir", type=str, default="./processed_data", help="Directory for processed data cache")
    parser.add_argument("--show_languages", action="store_true", help="Show supported languages and datasets")
    
    args = parser.parse_args()
    
    # 如果用户要求显示支持的语言
    if args.show_languages:
        print_language_support_info()
        exit(0)
    
    # 验证语言和数据集组合
    if not validate_language_dataset_combo(args.lang, args.dataset):
        print(f"❌ Error: Dataset '{args.dataset}' is not supported for language '{args.lang}'")
        print("Use --show_languages to see supported combinations")
        exit(1)
    
    # 返回参数
    return args
