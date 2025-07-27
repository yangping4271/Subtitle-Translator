import logging
import logging.handlers
import os
import sys
from pathlib import Path
import queue
import threading
from typing import Optional
import time

# 路径 - 日志文件放在cli目录下
ROOT_PATH = Path(__file__).parent.parent.parent  # 从src/subtitle_translator/回到cli目录
LOG_PATH = ROOT_PATH / "logs"
LOG_FILE = str(LOG_PATH / 'app.log')

# 全局日志队列
log_queue = queue.Queue()
queue_handler = None
_queue_listener = None

# 检查命令行参数中是否有-d或--debug
def is_debug_mode():
    return '-d' in sys.argv or '--debug' in sys.argv or os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes')

# 日志配置
# LOG_LEVEL = logging.DEBUG if is_debug_mode() else logging.INFO # 这一行将被移除

class ColoredFormatter(logging.Formatter):
    """带颜色和emoji的日志格式化器"""
    
    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    # 日志级别对应的emoji
    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': '📋',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    def __init__(self, use_color=True, use_emoji=True):
        self.use_color = use_color
        self.use_emoji = use_emoji
        super().__init__()
    
    def format(self, record):
        # 获取基础时间格式
        time_str = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        
        # 添加emoji（如果启用）
        emoji = self.EMOJIS.get(record.levelname, '') if self.use_emoji else ''
        
        # 获取模块名简化版
        module_name = self._get_simplified_module_name(record.name)
        
        # 构建日志消息
        if self.use_color:
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            formatted_msg = f"{time_str} {emoji} [{color}{module_name}{reset}] {record.getMessage()}"
        else:
            formatted_msg = f"{time_str} {emoji} [{module_name}] {record.getMessage()}"
        
        return formatted_msg
    
    def _get_simplified_module_name(self, name):
        """简化模块名显示"""
        name_mapping = {
            '__main__': '主程序',
            'split_by_llm': '智能断句',
            'subtitle_merger': '断句合并',
            'subtitle_summarizer': '内容总结',
            'subtitle_optimizer': '翻译优化',
            'subtitle_aligner': '字幕对齐',
            'subtitle_data': '数据处理',
            'json_repair': 'JSON修复'
        }
        return name_mapping.get(name, name)

class ProgressLogger:
    """进度日志工具"""
    
    def __init__(self, logger, total_steps, task_name="任务"):
        self.logger = logger
        self.total_steps = total_steps
        self.task_name = task_name
        self.current_step = 0
        self.start_time = time.time()
    
    def update(self, step=None, message=""):
        """更新进度"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        percentage = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        
        # 生成进度条
        bar_length = 20
        filled_length = int(bar_length * self.current_step / self.total_steps)
        bar = '█' * filled_length + '▒' * (bar_length - filled_length)
        
        # 预估剩余时间
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            eta_str = f" (预计剩余: {eta:.0f}秒)" if eta > 1 else ""
        else:
            eta_str = ""
        
        progress_msg = f"🚀 {self.task_name} [{bar}] {percentage:.1f}% ({self.current_step}/{self.total_steps}){eta_str}"
        if message:
            progress_msg += f" - {message}"
            
        self.logger.info(progress_msg)
    
    def complete(self, message="任务完成"):
        """标记任务完成"""
        elapsed = time.time() - self.start_time
        self.logger.info(f"✅ {self.task_name}完成！总耗时: {elapsed:.1f}秒 - {message}")

class QueueListenerHandler(logging.handlers.QueueHandler):
    """
    将日志记录放入队列的处理器
    """
    def __init__(self, queue, level):
        super().__init__(queue)
        self._queue_listener = None
        self.level = level # 添加level属性
        
    def start_listener(self):
        if self._queue_listener is None:
            handlers = self._create_handlers()
            self._queue_listener = logging.handlers.QueueListener(
                self.queue,
                *handlers,
                respect_handler_level=True
            )
            self._queue_listener.start()
            
    def _create_handlers(self):
        # 只创建文件处理器，不创建控制台处理器以避免与print重复输出
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            LOG_FILE,
            mode='w',  # 使用写入模式，覆盖旧文件
            encoding='utf-8'
        )
        file_formatter = ColoredFormatter(use_color=False, use_emoji=True)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(self.level)
        
        return [file_handler]

def setup_logger(name: str, 
                debug_mode: bool = False,
                log_fmt: str = '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                datefmt: str = '%Y-%m-%d %H:%M:%S') -> logging.Logger:
    """
    创建并配置一个日志记录器。

    参数：
    - name: 日志记录器的名称
    - debug_mode: 是否启用调试模式
    - log_fmt: 日志格式字符串
    - datefmt: 时间格式字符串
    """
    global queue_handler
    
    level = logging.DEBUG if debug_mode else logging.INFO
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 防止logger传播到根logger，避免重复输出
    logger.propagate = False
    
    # 检查是否已经有处理器，如果有则直接返回，避免重复配置
    if logger.handlers:
        return logger
    
    # 创建队列处理器（如果还没有创建）
    if queue_handler is None:
        queue_handler = QueueListenerHandler(log_queue, level)
        queue_handler.start_listener()
    
    logger.addHandler(queue_handler)
    
    # 设置特定库的日志级别为ERROR以减少日志噪音
    error_loggers = ["urllib3", "requests", "openai", "httpx", "httpcore", "ssl", "certifi"]
    for lib in error_loggers:
        logging.getLogger(lib).setLevel(logging.ERROR)
    
    return logger

def create_progress_logger(logger, total_steps, task_name="任务"):
    """创建进度日志器的便捷函数"""
    return ProgressLogger(logger, total_steps, task_name)

def log_section_start(logger, section_name, emoji="🔧"):
    """记录节开始"""
    logger.info(f"\n{emoji} {section_name} 开始")

def log_section_end(logger, section_name, elapsed_time=None, emoji="✅"):
    """记录节结束"""
    time_info = f" (耗时: {elapsed_time:.1f}秒)" if elapsed_time else ""
    logger.info(f"{emoji} {section_name} 完成{time_info}\n")

def log_stats(logger, stats_dict, title="统计信息"):
    """记录统计信息"""
    logger.info(f"📊 {title}:")
    for key, value in stats_dict.items():
        logger.info(f"   {key}: {value}")

def shutdown_logging():
    """
    关闭日志系统，确保所有日志都被写入
    """
    global queue_handler
    if queue_handler and queue_handler._queue_listener:
        # 等待队列中的所有日志都被处理
        log_queue.join()
        queue_handler._queue_listener.stop()
        queue_handler = None
        
# 确保程序退出时正确关闭日志
import atexit
atexit.register(shutdown_logging)
