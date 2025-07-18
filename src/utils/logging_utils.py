import os
import logging

def setup_logging(log_dir="./logs", run_name="training"):
    """
    设置日志记录配置
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    # 创建日志文件
    log_file = os.path.join(log_dir, f"{run_name}.log")
    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    # 返回日志记录器
    return logging.getLogger(__name__) 