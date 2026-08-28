# logging_config.py
"""
日志配置模块

功能：
- 彩色控制台输出
- 可选文件日志记录
- 进度条支持
- 自定义日志名称

使用方法：
    from logging_config import setup_logging, log_progress
    
    logger = setup_logging('my_app', log_dir='./logs')
    
    # 进度条
    for i in range(101):
        log_progress(logger, i, 100, '训练中')
        time.sleep(0.05)
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """带颜色的格式化器"""
    
    COLORS = {
        'DEBUG': '\033[94m',      # 蓝色
        'INFO': '\033[92m',       # 绿色
        'WARNING': '\033[93m',    # 黄色
        'ERROR': '\033[91m',      # 红色
        'CRITICAL': '\033[1;91m', # 加粗红色
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']
        
        # 进度条不显示级别
        if getattr(record, 'is_progress', False):
            return record.getMessage()
        
        msg = f"{color}{record.levelname:8}{reset} | {record.getMessage()}"
        time_str = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        return f"{time_str} | {msg}"


class ProgressHandler(logging.StreamHandler):
    """支持进度条的处理器"""
    
    def __init__(self, stream=sys.stdout):
        super().__init__(stream)
        self.last_progress = None
    
    def emit(self, record):
        if getattr(record, 'is_progress', False):
            msg = self.format(record)
            # 覆盖上一行
            if self.last_progress:
                self.stream.write('\r' + msg)
            else:
                self.stream.write(msg)
            self.stream.flush()
            self.last_progress = msg
        else:
            # 普通日志，先换行清除进度条
            if self.last_progress:
                self.stream.write('\n')
                self.last_progress = None
            super().emit(record)


def setup_logging(name='app', 
                  log_dir=None,
                  log_name=None,
                  level=logging.INFO,
                  max_mb=10,
                  backup=5):
    """
    配置日志
    
    Parameters:
    -----------
    name : str
        日志名称
    log_dir : str, optional
        日志目录，None则不写入文件
    log_name : str, optional
        自定义日志文件名前缀
    level : int
        日志级别
    max_mb : int
        单个日志文件最大大小(MB)
    backup : int
        保留备份数量
    
    Returns:
    --------
    logger : logging.Logger
    """
    logger = logging.getLogger(name)
    # logger.handlers = []
    if not logger.handlers:
        # 控制台输出
        console = ProgressHandler(sys.stdout)
        console.setFormatter(ColoredFormatter())
        logger.addHandler(console)
        logger.setLevel(level)  # 只在首次设置级别
        
    # 文件输出（可选，无进度条）
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
        if log_name:
            filename = f"{log_name}.log"
        else:
            filename = f"{name}_{datetime.now():%Y%m%d}.log"
        
        log_file = os.path.join(log_dir, filename)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_mb * 1024 * 1024,
            backupCount=backup,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    
    return logger


def log_progress(logger, current, total, prefix='', suffix=''):
    """
    打印进度条
    
    Parameters:
    -----------
    logger : logging.Logger
    current : int
        当前进度
    total : int
        总进度
    prefix : str
        前缀文字
    suffix : str
        后缀文字
    """
    percent = 100 * current / total
    filled = int(50 * current / total)
    bar = '█' * filled + '-' * (50 - filled)
    
    msg = f"{prefix} |{bar}| {percent:.1f}% {suffix}"
    
    # 创建进度记录
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        '',
        0,
        msg,
        (),
        None
    )
    record.is_progress = True
    
    # 直接发送给控制台handler
    for handler in logger.handlers:
        if isinstance(handler, ProgressHandler):
            handler.emit(record)
    
    # 完成时换行
    if current >= total:
        for handler in logger.handlers:
            if isinstance(handler, ProgressHandler):
                handler.stream.write('\n')
                handler.last_progress = None


# # 使用示例
# if __name__ == '__main__':
#     import time
    
#     # 仅控制台
#     logger = setup_logging('demo')
    
#     logger.info("开始训练")
    
#     # 进度条
#     for i in range(101):
#         log_progress(logger, i, 100, prefix='训练中', suffix=f'第{i}轮')
#         time.sleep(0.02)
    
#     logger.info("训练完成")
    
#     # 带文件的
#     logger2 = setup_logging('train', log_dir='./logs', log_name='train_v1')
#     logger2.info("日志也会写入文件")