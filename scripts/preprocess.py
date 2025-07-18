import argparse
import sys
import os

# 将项目根目录添加到Python路径中，以便导入src模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_and_preprocess_data
import config

def main():
    parser = argparse.ArgumentParser(description="Preprocess and cache a dataset for NER.")
    parser.add_argument("--dataset", type=str, default=config.DEFAULT_DATASET_NAME, help="Dataset name (e.g., conll2003)")
    parser.add_argument("--lang", type=str, default="en", help="Language/config for the dataset (e.g., 'en', 'zh')")
    parser.add_argument("--max_len", type=int, default=config.MAX_LEN, help="Max sequence length")
    parser.add_argument("--min_freq", type=int, default=1, help="Min frequency for vocab")
    parser.add_argument("--force_preprocess", action="store_true", help="Force preprocessing even if cache exists.")
    args = parser.parse_args()

    print(f"--- Starting preprocessing for dataset: {args.dataset} (lang: {args.lang}) ---")
    datasets, label_names, id2label, label2id, vocab = load_and_preprocess_data(
        dataset_name=args.dataset,
        lang=args.lang,
        processed_data_dir=config.PROCESSED_DATA_DIR,
        max_len=args.max_len,
        min_freq=args.min_freq,
        force_preprocess=args.force_preprocess
    )
    print(f"--- Preprocessing finished. Cached data is in '{config.PROCESSED_DATA_DIR}' ---")
    print(f"Labels: {label_names}")
    print(f"Vocab size: {len(vocab)}")
    for split in datasets:
        print(f"{split} samples: {len(datasets[split])}")

if __name__ == "__main__":
    main()
