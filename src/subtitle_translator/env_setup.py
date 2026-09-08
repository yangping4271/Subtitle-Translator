"""
环境配置管理模块 - 负责环境变量加载和配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from .exceptions import ConfigurationError
from .translation_core.config import (
    resolve_configured_models,
    validate_api_configuration,
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

    API 和模型均必须手动配置；缺失时提示运行 'translate init'。
    """
    global _env_loaded, logger

    if _env_loaded:
        return

    required_vars = ['OPENAI_BASE_URL', 'OPENAI_API_KEY']

    # 先加载配置文件（不覆盖已有的环境变量）
    env_path = _get_config_path()
    if env_path.is_file():
        load_dotenv(env_path, verbose=False)

    _env_loaded = True

    if logger is None:
        from .logger import setup_logger
        logger = setup_logger(__name__)

    missing_vars = [v for v in required_vars if not os.environ.get(v)]
    _, split_model, translation_model = resolve_configured_models(os.environ)
    if not split_model:
        missing_vars.append("SPLIT_MODEL")
    if not translation_model:
        missing_vars.append("TRANSLATION_MODEL")

    openai_base_url = os.environ.get("OPENAI_BASE_URL", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_base_url and openai_api_key:
        try:
            validate_api_configuration(openai_base_url, openai_api_key)
        except ValueError as exc:
            from rich import print as rprint

            rprint(f"[red]❌ 配置无效:[/red] {exc}")
            raise ConfigurationError(str(exc)) from exc

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
        rprint("      [dim]OPENAI_API_KEY=your-api-key-here[/dim]")
        rprint("      [dim]SPLIT_MODEL=your-split-model[/dim]")
        rprint("      [dim]TRANSLATION_MODEL=your-translation-model[/dim]")
        rprint("      [dim]# 也可只设 LLM_MODEL 作为断句和翻译的共用模型[/dim]")
        rprint()
        raise ConfigurationError("缺少必需的配置项，请运行 'translate init' 初始化配置")

    try:
        validate_model_configuration(split_model, translation_model)
    except ValueError as exc:
        from rich import print as rprint

        rprint(f"[red]❌ 配置无效:[/red] {exc}")
        raise ConfigurationError(str(exc)) from exc
