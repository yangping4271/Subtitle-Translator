"""
环境配置管理模块 - 负责环境变量加载和配置
"""
import os
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# 应用名称，用于配置文件目录
APP_NAME = "subtitle_translator"

# 全局变量，用于跟踪环境是否已经加载
_env_loaded = False
logger = None


def setup_environment():
    """
    智能加载 .env 文件，解决在不同目录下运行命令的环境变量问题。
    加载顺序 (后者覆盖前者):
    1. 用户全局配置文件 (~/.config/subtitle_translator/.env)
    2. 项目配置文件 (从当前目录向上找到的第一个 .env)
    
    特殊功能：
    - 如果全局配置不存在，但找到项目配置，会自动复制项目配置作为全局配置
    - 使用标准的 .config 目录存储全局配置
    """
    global _env_loaded, logger
    
    # 如果已经加载过环境配置，直接返回
    if _env_loaded:
        return
    
    env_loaded = False
    
    # 准备路径 - 使用标准的 .config 目录
    app_dir = Path.home() / ".config" / APP_NAME
    user_env_path = app_dir / ".env"
    
    # 确保目录存在
    app_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找项目本地的 .env 文件
    project_env_path_str = find_dotenv(usecwd=True)
    project_env_path = Path(project_env_path_str) if project_env_path_str else None
    
    # 🎯 智能配置复制：如果全局配置不存在但项目配置存在，自动复制
    config_copied = False
    if not user_env_path.is_file() and project_env_path and project_env_path.is_file():
        try:
            import shutil
            shutil.copy2(project_env_path, user_env_path)
            config_copied = True
        except Exception as e:
            print(f"⚠️  复制配置文件失败: {e}")

    # 1. 加载用户全局配置文件 (适用于已安装的应用)
    if user_env_path.is_file():
        load_dotenv(user_env_path, verbose=False)
        env_loaded = True
        
    # 2. 加载项目本地的 .env 文件 (方便开发，并可覆盖全局配置)
    if project_env_path and project_env_path.is_file():
        load_dotenv(project_env_path, verbose=False, override=True)
        env_loaded = True
    
    # 标记环境已加载
    _env_loaded = True
    
    # 初始化logger（需要在环境变量加载后进行）
    if logger is None:
        # 检测debug模式：检查命令行参数和环境变量
        debug_mode = ('-d' in sys.argv or '--debug' in sys.argv or 
                     os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes'))
        
        from .logger import setup_logger
        logger = setup_logger(__name__, debug_mode=debug_mode)
        
        # 只在需要提醒用户或出现问题时输出日志信息
        if config_copied:
            logger.info(f"✅ 首次运行检测到项目配置文件，已自动复制到全局配置:")
            logger.info(f"   源文件: {project_env_path}")
            logger.info(f"   目标文件: {user_env_path}")
            logger.info(f"   现在你可以在任意目录下运行 subtitle-translate 命令！")
        elif not env_loaded:
            logger.warning(
                f"未找到任何 .env 文件。程序将依赖于系统环境变量。\n"
                f"如需通过文件配置，请在项目根目录或用户配置目录 "
                f"({app_dir}) 中创建一个 .env 文件。"
            )
            
            # 检查关键环境变量是否存在
            required_vars = ['OPENAI_BASE_URL', 'OPENAI_API_KEY', 'LLM_MODEL']
            missing_vars = []
            for var in required_vars:
                if not os.environ.get(var):
                    missing_vars.append(var)
            
            if missing_vars:
                logger.error(f"缺少必需的环境变量: {', '.join(missing_vars)}")
                logger.error("请运行 'subtitle-translate init' 来配置API密钥，或设置相应的环境变量。")
                sys.exit(1)


def get_app_config_dir() -> Path:
    """获取应用配置目录"""
    return Path.home() / ".config" / APP_NAME


def get_global_env_path() -> Path:
    """获取全局环境配置文件路径"""
    return get_app_config_dir() / ".env"


class OpenAIAPIError(Exception):
    """OpenAI API 相关错误"""
    pass 