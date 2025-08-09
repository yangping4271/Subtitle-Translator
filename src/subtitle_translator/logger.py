import logging
import logging.handlers
import os
import sys
from pathlib import Path
import queue
import threading
from typing import Optional
import time

# 智能日志路径选择
def _get_log_path():
    """智能选择日志路径：开发模式使用项目目录，生产模式使用用户目录"""
    # 项目根目录路径
    project_root = Path(__file__).parent.parent.parent
    project_log_path = project_root / "logs"
    
    # 检查是否在开发环境（项目目录下有pyproject.toml等标志文件）
    if (project_root / "pyproject.toml").exists() and (project_root / "src").exists():
        # 开发模式：使用项目目录
        return project_log_path
    else:
        # 生产模式：使用用户目录
        user_log_path = Path.home() / ".local" / "share" / "subtitle-translator" / "logs"
        return user_log_path

LOG_PATH = _get_log_path()
LOG_FILE = str(LOG_PATH / 'app.log')

# 全局日志队列
log_queue = queue.Queue()
queue_handler = None
_queue_listener = None

# 全局debug状态管理
_global_debug_mode = False
_initialized_loggers = []  # 记录所有已创建的logger名称

def get_log_file_path():
    """获取当前使用的日志文件路径"""
    return LOG_FILE

def configure_all_loggers(debug_mode: bool):
    """
    全局配置所有已创建的logger的debug级别
    
    Args:
        debug_mode: 是否启用debug模式
    """
    global _global_debug_mode, queue_handler
    
    # 更新全局debug状态
    _global_debug_mode = debug_mode
    
    # 如果队列处理器已经存在，更新其级别
    if queue_handler is not None:
        target_level = logging.DEBUG if debug_mode else logging.INFO
        queue_handler.update_level(target_level)
        
        # 同时更新所有已创建的logger
        for logger_name in _initialized_loggers:
            logger_instance = logging.getLogger(logger_name)
            logger_instance.setLevel(target_level)

def get_log_mode_info():
    """获取日志模式信息（开发模式或生产模式）"""
    project_root = Path(__file__).parent.parent.parent
    if (project_root / "pyproject.toml").exists() and (project_root / "src").exists():
        return "开发模式", "项目目录"
    else:
        return "生产模式", "用户目录"

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
        
        # 为DEBUG信息添加缩进，提高可读性
        if record.levelname == 'DEBUG':
            # 检查消息是否以特定格式开头（如详细数据），如果是则添加缩进
            message = record.getMessage()
            if any(message.startswith(prefix) for prefix in ['   ', '  ID', '  原文:', '  译文:', '  优化:']):
                # 已经有缩进的消息，保持原格式
                pass
            elif message.strip() and not any(message.startswith(emoji) for emoji in ['🔍', '📋', '⚠️', '❌', '🚨']):
                # 为详细调试信息添加缩进
                lines = formatted_msg.split('\n')
                if len(lines) > 1:
                    # 多行消息，从第二行开始缩进
                    indented_lines = [lines[0]] + ['    ' + line for line in lines[1:]]
                    formatted_msg = '\n'.join(indented_lines)
                else:
                    # 单行消息，检查是否需要缩进
                    if not message.startswith(('🔍', '📤', '📥', '✅', '🔧')):
                        # 不是主要步骤消息，添加缩进表示详细信息
                        formatted_msg = formatted_msg.replace(f'] {message}', f']     {message}')
        
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
    支持动态调整日志级别
    """
    def __init__(self, queue, level):
        super().__init__(queue)
        self._queue_listener = None
        self._file_handler = None
        self._current_level = level
        
    def update_level(self, new_level):
        """更新日志级别，如果级别发生变化则重新配置文件处理器"""
        if new_level != self._current_level:
            self._current_level = new_level
            if self._file_handler:
                self._file_handler.setLevel(new_level)
        
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
        self._file_handler = logging.FileHandler(
            LOG_FILE,
            mode='w',  # 使用写入模式，覆盖旧文件
            encoding='utf-8'
        )
        file_formatter = ColoredFormatter(use_color=False, use_emoji=True)
        self._file_handler.setFormatter(file_formatter)
        self._file_handler.setLevel(self._current_level)
        
        return [self._file_handler]

def setup_logger(name: str, 
                debug_mode: bool = None,
                log_fmt: str = '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                datefmt: str = '%Y-%m-%d %H:%M:%S') -> logging.Logger:
    """
    创建并配置一个日志记录器。
    支持全局debug状态管理和延迟配置。

    参数：
    - name: 日志记录器的名称
    - debug_mode: 是否启用调试模式。如果为None，则使用全局debug状态
    - log_fmt: 日志格式字符串
    - datefmt: 时间格式字符串
    """
    global queue_handler, _global_debug_mode, _initialized_loggers
    
    # 记录这个logger，以便后续全局配置
    if name not in _initialized_loggers:
        _initialized_loggers.append(name)
    
    # 决定使用的debug模式：优先使用传入参数，否则使用全局状态
    effective_debug_mode = debug_mode if debug_mode is not None else _global_debug_mode
    level = logging.DEBUG if effective_debug_mode else logging.INFO
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 防止logger传播到根logger，避免重复输出
    logger.propagate = False
    
    # 检查是否已经有处理器，如果有则直接返回，避免重复配置
    if logger.handlers:
        return logger
    
    # 创建或更新队列处理器
    if queue_handler is None:
        queue_handler = QueueListenerHandler(log_queue, level)
        queue_handler.start_listener()
    else:
        # 如果队列处理器已存在，更新其级别以支持更细粒度的日志
        # 使用更低的级别（DEBUG优先级更高）
        if level < queue_handler._current_level:
            queue_handler.update_level(level)
    
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
