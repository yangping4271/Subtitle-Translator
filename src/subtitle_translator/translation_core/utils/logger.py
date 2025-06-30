import logging
import logging.handlers
import os
import sys
from pathlib import Path
import queue
import threading
from typing import Optional

# 路径
ROOT_PATH = Path(__file__).parent.parent
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
        formatter = logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.level) # 使用实例的level
        
        # 文件处理器
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            LOG_FILE,
            mode='w',  # 使用写入模式，覆盖旧文件
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.level) # 使用实例的level
        
        return [console_handler, file_handler]

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
    
    # 清除现有处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建队列处理器（如果还没有创建）
    if queue_handler is None:
        queue_handler = QueueListenerHandler(log_queue, level) # 传入level
        queue_handler.start_listener()
    
    logger.addHandler(queue_handler)
    
    # 设置特定库的日志级别为ERROR以减少日志噪音
    error_loggers = ["urllib3", "requests", "openai", "httpx", "httpcore", "ssl", "certifi"]
    for lib in error_loggers:
        logging.getLogger(lib).setLevel(logging.ERROR)
    
    return logger

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
