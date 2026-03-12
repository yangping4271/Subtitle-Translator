"""
版本信息工具模块
"""
import sys
from pathlib import Path
from typing import Dict, Optional
import importlib.metadata
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def get_project_version() -> str:
    """从pyproject.toml或已安装包获取项目版本"""
    try:
        # 首先尝试从已安装的包获取版本
        return importlib.metadata.version("subtitle-translator")
    except importlib.metadata.PackageNotFoundError:
        # 如果包未安装，尝试从pyproject.toml读取
        try:
            import tomli
            pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    data = tomli.load(f)
                    return data.get("project", {}).get("version", "Unknown")
        except ImportError:
            # 如果没有tomli，尝试简单的文本解析
            try:
                pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
                if pyproject_path.exists():
                    with open(pyproject_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("version ="):
                                # 解析 version = "0.2.5" 格式
                                return line.split("=", 1)[1].strip().strip('"\'')
            except Exception:
                pass
        return "Unknown"


def get_dependency_versions() -> Dict[str, str]:
    """获取主要依赖的版本信息"""
    dependencies = {
        "openai": "OpenAI 客户端",
        "typer": "命令行框架",
        "rich": "终端美化",
    }

    versions = {}
    for dep_name, desc in dependencies.items():
        try:
            version = importlib.metadata.version(dep_name)
            versions[dep_name] = {"version": version, "description": desc}
        except importlib.metadata.PackageNotFoundError:
            versions[dep_name] = {"version": "未安装", "description": desc}

    return versions


def get_simple_version_info() -> str:
    """获取简洁的版本信息字符串"""
    project_version = get_project_version()
    return f"Subtitle Translator v{project_version}"


def display_version_info(console: Optional[Console] = None) -> None:
    """显示格式化的版本信息"""
    if console is None:
        console = Console()
    
    # 获取版本信息
    project_version = get_project_version()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    dependencies = get_dependency_versions()
    
    # 创建主版本信息面板
    version_text = f"""
[bold cyan]Subtitle Translator[/bold cyan] v{project_version}
字幕翻译命令行工具

[dim]Python版本:[/dim] {python_version}
[dim]平台:[/dim] {sys.platform}
"""
    
    main_panel = Panel(
        version_text.strip(),
        title="🚀 版本信息",
        border_style="cyan"
    )
    
    # 创建依赖信息表格
    dep_table = Table(title="📦 主要依赖")
    dep_table.add_column("依赖", style="cyan")
    dep_table.add_column("版本", style="green")  
    dep_table.add_column("描述", style="dim")
    
    for dep_name, info in dependencies.items():
        dep_table.add_row(
            dep_name,
            info["version"],
            info["description"]
        )
    
    # 显示信息
    console.print()
    console.print(main_panel)
    console.print()
    console.print(dep_table)
    console.print()
    console.print("[dim]💡 获取更多帮助: translate --help[/dim]")


if __name__ == "__main__":
    display_version_info()