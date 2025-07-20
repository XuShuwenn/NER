import os 
import re #正则表达式
import matplotlib.pyplot as plt 

# 日志目录和输出目录
LOG_DIR = "logs"
OUT_DIR = "log_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# 正则匹配epoch日志行
pattern = re.compile(
    r"Epoch (\d+): Train Loss=([0-9.]+), Train Acc=([0-9.]+), Val Loss=([0-9.]+), Val Acc=([0-9.]+), Eval F1=([0-9.]+)"
)

# 解析日志文件
def parse_log(log_path):
    epochs, train_loss, val_loss, f1, train_acc, val_acc = [], [], [], [], [], []
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                train_acc.append(float(m.group(3)))
                val_loss.append(float(m.group(4)))
                val_acc.append(float(m.group(5)))
                f1.append(float(m.group(6)))
    return epochs, train_loss, val_loss, f1, train_acc, val_acc

# 绘制训练曲线
def plot_metrics(log_file, epochs, train_loss, val_loss, f1, train_acc, val_acc):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.plot(epochs, f1, label="Eval F1")
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"Training Curve: {os.path.basename(log_file)}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 创建对应的输出子目录
    rel_path = os.path.relpath(log_file, LOG_DIR)
    sub_dir = os.path.dirname(rel_path)
    if sub_dir:
        out_sub_dir = os.path.join(OUT_DIR, sub_dir)
        os.makedirs(out_sub_dir, exist_ok=True)
        out_path = os.path.join(out_sub_dir, os.path.basename(log_file) + ".png")
    else:
        out_path = os.path.join(OUT_DIR, os.path.basename(log_file) + ".png")
    
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

# 递归查找所有日志文件
def find_log_files(directory):
    log_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".log"):
                log_files.append(os.path.join(root, file))
    return log_files

# 主函数
def main():
    
    log_files = find_log_files(LOG_DIR)
    if not log_files:
        print("No log files found in logs/ and its subdirectories")
        return
    
    print(f"Found {len(log_files)} log files:")
    for log_file in log_files:
        print(f"  - {log_file}")
    
    for log_file in log_files:
        epochs, train_loss, val_loss, f1, train_acc, val_acc = parse_log(log_file)
        if epochs:
            plot_metrics(log_file, epochs, train_loss, val_loss, f1, train_acc, val_acc)
        else:
            print(f"No valid epoch data found in {log_file}")

if __name__ == "__main__":
    main() 