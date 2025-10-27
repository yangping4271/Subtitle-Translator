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

    Returns:
        项目根目录的 Path 对象
    """
    # 从当前文件位置开始向上查找
    current = Path(__file__).resolve().parent

    while current != current.parent:
        # 检查是否包含项目标识文件
        if (current / ".git").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # 如果找不到，返回当前工作目录
    return Path.cwd()


def setup_environment(allow_missing_config=False):
    """
    从项目根目录加载 .env 文件

    项目根目录定义：包含 .git 或 pyproject.toml 的目录
    如果找不到 .env 文件，会尝试使用系统环境变量

    Args:
        allow_missing_config: 是否允许缺少配置（用于特殊场景）
    """
    global _env_loaded, logger

    # 如果已经加载过环境配置，直接返回
    if _env_loaded:
        return

    env_loaded = False

    # 查找项目根目录
    project_root = _find_project_root()
    env_path = project_root / ".env"

    # 加载项目根目录的 .env 文件
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
            print(f"[yellow]⚠️  未找到配置文件[/yellow]")
            print(f"   项目根目录: [cyan]{project_root}[/cyan]")
            print(f"   配置文件: [cyan]{env_path}[/cyan]")
            print()

            # 检查关键环境变量是否存在
            required_vars = ['OPENAI_BASE_URL', 'OPENAI_API_KEY', 'LLM_MODEL']
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
                    print(f"[bold blue]💡 解决方法:[/bold blue]")
                    print(f"   1. 在项目根目录创建 .env 文件")
                    print(f"      [dim]位置: {project_root}/.env[/dim]")
                    print()
                    print(f"   2. 参考配置示例:")
                    print(f"      [dim]cp {project_root}/env.example {project_root}/.env[/dim]")
                    print()
                    print(f"   3. 编辑 .env 文件，填入你的配置:")
                    print(f"      [dim]OPENAI_BASE_URL=https://api.openai.com/v1[/dim]")
                    print(f"      [dim]OPENAI_API_KEY=your-api-key-here[/dim]")
                    print(f"      [dim]SPLIT_MODEL=gpt-4o-mini[/dim]")
                    print(f"      [dim]TRANSLATION_MODEL=gpt-4o[/dim]")
                    print(f"      [dim]SUMMARY_MODEL=gpt-4o-mini[/dim]")
                    print(f"      [dim]LLM_MODEL=gpt-4o-mini[/dim]")
                    print()
                    sys.exit(1)


class OpenAIAPIError(Exception):
    """OpenAI API 相关错误"""
    pass
