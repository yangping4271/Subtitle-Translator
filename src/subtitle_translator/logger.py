import atexit
import logging
import logging.handlers
from pathlib import Path
import queue
from typing import Optional, Tuple


def _find_project_root() -> Optional[Path]:
    """
    查找项目根目录（包含 pyproject.toml 和 src/subtitle_translator 的目录）

    Returns:
        项目根目录的 Path 对象，如果不在项目目录内则返回 None
    """
    try:
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists() and (current / "src" / "subtitle_translator").exists():
                return current
            current = current.parent
    except Exception:
        pass
    return None


def _get_log_path() -> Path:
    """
    智能选择日志路径：
    - 开发模式（当前工作目录在项目内）: 使用项目目录
    - 全局工具模式: 使用用户目录
    """
    project_root = _find_project_root()
    if project_root:
        return project_root / "logs"
    return Path.home() / ".local" / "share" / "subtitle-translator" / "logs"


LOG_PATH = _get_log_path()
LOG_FILE = str(LOG_PATH / 'app.log')

# 全局日志队列
log_queue = queue.Queue()
queue_handler = None
_queue_listener = None


def get_log_file_path() -> str:
    """获取当前使用的日志文件路径"""
    return LOG_FILE


def get_log_mode_info() -> Tuple[str, str]:
    """获取日志模式信息（开发模式或生产模式）"""
    if _find_project_root():
        return "开发模式", "项目目录"
    return "生产模式", "用户目录"

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
            'response_parser': '响应解析'
        }
        return name_mapping.get(name, name)

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
                self._file_handler.setLevel(logging.DEBUG)
        
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
            mode='w',  # 使用覆盖模式，每个新任务覆盖旧日志
            encoding='utf-8'
        )
        file_formatter = ColoredFormatter(use_color=False, use_emoji=True)
        self._file_handler.setFormatter(file_formatter)
        # 文件处理器使用 DEBUG 级别，记录所有详细信息
        self._file_handler.setLevel(logging.DEBUG)

        return [self._file_handler]

def setup_logger(name: str,
                log_fmt: str = '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                datefmt: str = '%Y-%m-%d %H:%M:%S') -> logging.Logger:
    """
    创建并配置一个日志记录器。
    logger 允许 DEBUG 及以上级别进入队列，由文件处理器统一写入日志。
    终端输出仍由各处的 print 控制，避免与日志重复输出。

    参数：
    - name: 日志记录器的名称
    - log_fmt: 日志格式字符串
    - datefmt: 时间格式字符串
    """
    global queue_handler

    level = logging.DEBUG

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

    logger.addHandler(queue_handler)

    # 设置特定库的日志级别为ERROR以减少日志噪音
    error_loggers = ["urllib3", "requests", "openai", "httpx", "httpcore", "ssl", "certifi"]
    for lib in error_loggers:
        logging.getLogger(lib).setLevel(logging.ERROR)

    return logger

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
atexit.register(shutdown_logging)
