"""
环境配置管理模块 - 负责环境变量加载和配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from .exceptions import ConfigurationError
from .translation_core.config import (
    resolve_configured_model,
    validate_model_configuration,
)

_env_loaded = False
logger = None


def _get_config_path() -> Path:
    """配置文件路径：~/.config/subtitle-translator/.env"""
    return Path.home() / ".config" / "subtitle-translator" / ".env"


def setup_environment(allow_missing_config=False):
    """
    加载环境配置。

    配置来源（按优先级）：
    1. 已有的环境变量
    2. ~/.config/subtitle-translator/.env

    API 与模型均必须手动配置；API Key 可留空，用于本地或无鉴权端点。
    """
    global _env_loaded, logger

    if _env_loaded:
        return

    required_vars = ['OPENAI_BASE_URL']

    # 先加载配置文件（不覆盖已有的环境变量）
    env_path = _get_config_path()
    if env_path.is_file():
        load_dotenv(env_path, verbose=False)

    _env_loaded = True

    if logger is None:
        from .logger import setup_logger
        logger = setup_logger(__name__)

    missing_vars = [v for v in required_vars if not os.environ.get(v)]
    llm_model = resolve_configured_model(os.environ)
    if not llm_model:
        missing_vars.append("LLM_MODEL")

    if missing_vars:
        if allow_missing_config:
            logger.warning(f"缺少必需的环境变量: {', '.join(missing_vars)}")
            logger.warning("程序将在配置模式下运行。")
            return
        from rich import print as rprint

        rprint("[red]❌ 缺少必需的配置项:[/red]")
        for var in missing_vars:
            rprint(f"   • {var}")
        rprint()
        rprint("[bold blue]💡 快速开始:[/bold blue]")
        rprint("   [bold]1. 运行初始化命令[/bold]")
        rprint("      [green]translate init[/green]")
        rprint()
        rprint("   [bold]2. 或手动创建配置文件[/bold]")
        rprint(f"      [dim]位置: {env_path}[/dim]")
        rprint()
        rprint("   [bold]配置示例:[/bold]")
        rprint("      [dim]OPENAI_BASE_URL=https://api.openai.com/v1[/dim]")
        rprint("      [dim]OPENAI_API_KEY=your-api-key-here  # 无鉴权端点可留空[/dim]")
        rprint("      [dim]LLM_MODEL=your-model[/dim]")
        rprint()
        raise ConfigurationError("缺少必需的配置项，请运行 'translate init' 初始化配置")

    try:
        validate_model_configuration(llm_model)
    except ValueError as exc:
        from rich import print as rprint

        rprint(f"[red]❌ 配置无效:[/red] {exc}")
        raise ConfigurationError(str(exc)) from exc
