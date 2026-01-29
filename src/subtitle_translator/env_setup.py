"""
环境配置管理模块 - 负责环境变量加载和配置
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 全局变量，用于跟踪环境是否已经加载
_env_loaded = False
logger = None


def _find_project_root() -> Path:
    """
    查找项目根目录（包含 .git 或 pyproject.toml 的目录）

    仅用于开发模式检测。

    Returns:
        项目根目录的 Path 对象，如果找不到返回当前工作目录
    """
    # 从当前工作目录开始向上查找（用于开发模式）
    current = Path.cwd()
    while current != current.parent:
        if (current / ".git").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent

    return Path.cwd()


def _get_config_path() -> Path:
    """
    获取配置文件路径

    优先级：
    1. 开发模式：项目根目录的 .env（如果当前在项目目录下）
    2. 全局模式：~/.config/subtitle-translator/.env

    Returns:
        配置文件路径
    """
    # 检查是否在项目目录下（开发模式）
    project_root = _find_project_root()
    if (project_root / "pyproject.toml").exists() and (project_root / "src").exists():
        # 开发模式：使用项目目录的 .env
        return project_root / ".env"

    # 全局模式：使用 ~/.config/subtitle-translator/.env
    config_dir = Path.home() / ".config" / "subtitle-translator"
    return config_dir / ".env"


def setup_environment(allow_missing_config=False):
    """
    加载环境配置

    配置文件位置：
    1. 开发模式：项目根目录/.env（当前在项目目录下时）
    2. 全局模式：~/.config/subtitle-translator/.env

    如果配置文件不存在，会提示用户运行 'translate init' 初始化。

    Args:
        allow_missing_config: 是否允许缺少配置（用于特殊场景）
    """
    global _env_loaded, logger

    # 如果已经加载过环境配置，直接返回
    if _env_loaded:
        return

    env_loaded = False

    # 获取配置文件路径
    env_path = _get_config_path()

    # 尝试加载配置文件
    if env_path.is_file():
        load_dotenv(env_path, verbose=False)
        env_loaded = True

    # 标记环境已加载
    _env_loaded = True

    # 初始化logger（需要在环境变量加载后进行）
    if logger is None:
        from .logger import setup_logger
        logger = setup_logger(__name__)

        # 只在需要提醒用户或出现问题时输出日志信息
        if not env_loaded:
            from rich import print

            # 判断是否是开发模式
            is_dev_mode = env_path.parent.name != "subtitle-translator"

            print(f"[yellow]⚠️  未找到配置文件[/yellow]")
            print(f"   配置文件: [cyan]{env_path}[/cyan]")
            print()

            # 检查关键环境变量是否存在
            required_vars = ['OPENAI_BASE_URL', 'OPENAI_API_KEY']
            missing_vars = []
            for var in required_vars:
                if not os.environ.get(var):
                    missing_vars.append(var)

            if missing_vars:
                if allow_missing_config:
                    logger.warning(f"缺少必需的环境变量: {', '.join(missing_vars)}")
                    logger.warning("程序将在配置模式下运行。")
                else:
                    print(f"[red]❌ 缺少必需的配置项:[/red]")
                    for var in missing_vars:
                        print(f"   • {var}")
                    print()
                    print(f"[bold blue]💡 快速开始:[/bold blue]")
                    if is_dev_mode:
                        print(f"   [bold]开发模式[/bold] - 在项目根目录创建 .env 文件")
                        print(f"   [dim]位置: {env_path}[/dim]")
                    else:
                        print(f"   [bold]1. 运行初始化命令[/bold]")
                        print(f"      [green]translate init[/green]")
                        print()
                        print(f"   [bold]2. 或手动创建配置文件[/bold]")
                        print(f"      [dim]位置: {env_path}[/dim]")
                    print()
                    print(f"   [bold]配置示例:[/bold]")
                    print(f"      [dim]OPENAI_BASE_URL=https://api.openai.com/v1[/dim]")
                    print(f"      [dim]OPENAI_API_KEY=your-api-key-here[/dim]")
                    print(f"      [dim]SPLIT_MODEL=gpt-4o-mini[/dim]")
                    print(f"      [dim]TRANSLATION_MODEL=gpt-4o[/dim]")
                    print()
                    sys.exit(1)
