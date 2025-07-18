import argparse
import os
import torch
import logging
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler, AutoConfig
from seqeval.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from datasets import IterableDataset, IterableDatasetDict


from src.data_loader import load_and_preprocess_data, NERDataset, load_pretrained_embeddings
from src.models.lstm_ner import LSTMNER
from src.models.bert_ner import BertNER
from src.utils.arg_utils import get_train_args
from src.utils.logging_utils import setup_logging


def evaluate_epoch(model, dataloader, device, id2label, is_crf=False):
    """
    评估模型在验证集上的表现
    """
    # 设置模型为评估模式
    model.eval()
    # 初始化预测和真实标签列表
    all_predictions = []
    all_true_labels = []
    # 初始化损失和准确率
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    # 初始化损失函数
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    # 在评估模式下，不计算梯度
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # 将输入数据移动到设备上
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # 将掩码移动到设备上
            mask = batch["mask"].to(device)
            # 如果使用CRF模型
            if is_crf:
                # 计算损失
                loss = model(input_ids, labels=labels, mask=mask)
                # 预测标签
                outputs = model(input_ids, labels=None, mask=mask)
                # 将预测标签转换为列表
                predictions = outputs if isinstance(outputs, list) else outputs.tolist()
            else:
                outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                predictions = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()
            # 确保loss为标量，兼容多卡DataParallel
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            total_loss += loss.item() * input_ids.size(0)
            true_labels = labels.cpu().numpy().tolist()
            pred_labels = predictions
            for i in range(len(true_labels)):
                true_row = true_labels[i]
                pred_row = pred_labels[i]
                mask_row = mask[i].cpu().numpy()
                row_true, row_pred = [], []
                for t, p, m in zip(true_row, pred_row, mask_row):
                    if m and t != -100 and p != -100:
                        row_true.append(id2label[t])
                        row_pred.append(id2label[p])
                all_true_labels.append(row_true)
                all_predictions.append(row_pred)
                # 计算acc
                for t, p, m in zip(true_row, pred_row, mask_row):
                    if m and t != -100 and p != -100:
                        total_tokens += 1
                        if t == p:
                            correct_tokens += 1
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    acc = correct_tokens / total_tokens if total_tokens > 0 else 0
    return all_predictions, all_true_labels, avg_loss, acc


def main():

    args = get_train_args()
    # 获取训练参数（调用arg_utils.py中的get_train_args()函数，
    # 参数包括：模型类型、数据集、语言、模型名称或路径、嵌入路径、训练轮数、批量大小、最大序列长度、学习率、权重衰减、嵌入维度、隐藏维度、日志目录、处理后的数据目录）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.model_type}_{args.dataset}_{args.model_name_or_path.replace('/', '_')}"
    
    # Setup logging
    logger = setup_logging(args.log_dir, run_name)
    logger.info(f"Starting training with args: {vars(args)}")
    logger.info(f"Using device: {device}")

    # Data loading
    datasets, label_names, id2label, label2id, vocab = load_and_preprocess_data(
        dataset_name=args.dataset,
        lang=args.lang,
        processed_data_dir=args.processed_data_dir,
        max_len=args.max_len,
        min_freq=1,
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path
    )
    train_dataset = datasets["train"]
    val_dataset = datasets.get("validation", None)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size) if val_dataset else None

    # Model selection
    if args.model_type == "lstm-crf":
        if vocab is None:
            raise ValueError("Vocabulary is required for LSTM models")
        num_labels = len(label_names)
        embedding_matrix = None
        if args.embedding_path:
            embedding_matrix = load_pretrained_embeddings(vocab, args.embedding_path, args.embedding_dim)
        model = LSTMNER(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_labels=num_labels,
            embedding_matrix=embedding_matrix
        ).to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel!")
            model = torch.nn.DataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        is_crf = True
    else:  # bert
        hf_config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id
        )
        model = BertNER.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_names)
        )
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel!")
            model = torch.nn.DataParallel(model)
        optimizer = AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        is_crf = False

    num_training_steps = args.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    #Training Loop 一个训练循环
    best_eval_f1 = 0
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        total_tokens = 0
        correct_tokens = 0
        epoch_losses = []
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad()
            if is_crf:
                loss = model(input_ids, labels=labels, mask=mask)
                outputs = model(input_ids, labels=None, mask=mask)
                predictions = outputs if isinstance(outputs, list) else outputs.tolist()
            else:
                outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                predictions = outputs.logits.argmax(dim=-1)
            
            # 计算acc
            true_labels = labels.cpu().numpy()
            if is_crf:
                pred_labels = predictions
            else:
                pred_labels = predictions.cpu().numpy()
            mask_np = mask.cpu().numpy()
            for i in range(true_labels.shape[0]):
                for t, p, m in zip(true_labels[i], pred_labels[i], mask_np[i]):
                    if m:
                        total_tokens += 1
                        if t == p:
                            correct_tokens += 1
            global_step += 1
            
            # Log every 100 steps 每100步记录一次avg_loss
            if global_step % 100 == 0:
                avg_loss = sum(epoch_losses[-100:]) / len(epoch_losses[-100:])
                logger.info(f"Step {global_step}: Loss = {avg_loss:.4f}")

            # 确保loss为标量，兼容多卡DataParallel
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item() * input_ids.size(0)
            epoch_losses.append(loss.item())
        
        # 计算平均训练损失和准确率
        avg_train_loss = total_loss / total_tokens if total_tokens > 0 else 0
        train_acc = correct_tokens / total_tokens if total_tokens > 0 else 0

        # Validation验证
        if val_loader:
            predictions, true_labels, val_loss, val_acc = evaluate_epoch(model, val_loader, device, id2label, is_crf=is_crf)
            eval_f1 = f1_score(true_labels, predictions)
            eval_precision = precision_score(true_labels, predictions)
            eval_recall = recall_score(true_labels, predictions)
        else:
            val_loss = val_acc = eval_f1 = eval_precision = eval_recall = 0.0

        # Log epoch results
        logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                   f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
                   f"Eval F1={eval_f1:.4f}, Precision={eval_precision:.4f}, Recall={eval_recall:.4f}")
        
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Eval F1={eval_f1:.4f}")

    logger.info("Training completed!")


if __name__ == "__main__":
    # 主函数入口
    main()
