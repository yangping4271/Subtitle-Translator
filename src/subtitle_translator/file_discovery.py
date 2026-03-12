"""
文件发现和处理模块 - 负责查找、过滤和排序字幕文件
"""
import glob
import re
from pathlib import Path
from typing import List, Tuple, Optional

import typer
from rich import print


def remove_language_suffix(base_name: str) -> str:
    """移除文件名中的语言后缀"""
    language_suffixes = [
        r'\.zh$', r'\.zh-cn$', r'\.zh-tw$',  # 中文
        r'\.ja$', r'\.ko$', r'\.th$', r'\.vi$',  # 亚洲语言
        r'\.fr$', r'\.de$', r'\.es$', r'\.pt$', r'\.it$', r'\.ru$',  # 欧洲语言
        r'\.ar$', r'\.en$'  # 其他
    ]
    for suffix_pattern in language_suffixes:
        base_name = re.sub(suffix_pattern, '', base_name)
    return base_name


def natural_sort_key(s: str):
    """用于自然排序的key函数：将数字片段按整数比较，其他片段按不区分大小写的字符串比较"""
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.casefold() for p in parts]


def get_file_type_info(file_ext: str) -> Tuple[str, str]:
    """获取文件类型和处理方式"""
    if file_ext == '.srt':
        return "字幕文件", "直接翻译"
    return "未知类型", "未知"


def format_file_size(file_path: Path) -> str:
    """格式化文件大小显示"""
    try:
        file_size = file_path.stat().st_size
        if file_size < 1024:
            return f"{file_size} B"
        if file_size < 1024 * 1024:
            return f"{file_size / 1024:.1f} KB"
        return f"{file_size / (1024 * 1024):.1f} MB"
    except Exception:
        return "未知"


def get_batch_files(max_count: int, llm_model: Optional[str], input_dir: Path) -> List[Path]:
    """获取批量处理的文件列表"""
    # 只支持字幕文件
    patterns = ["*.srt"]

    # 确保input_dir是绝对路径
    input_dir = input_dir.resolve()

    # 查找所有 SRT 文件（使用绝对路径）
    srt_files = []
    for pattern in patterns:
        srt_files.extend(glob.glob(str(input_dir / pattern)))

    if not srt_files:
        print(f"[bold red]{input_dir} 目录中没有找到 SRT 字幕文件。[/bold red]")
        print("[dim]支持的格式：[/dim]")
        print("[dim]  • 字幕文件: .srt[/dim]")
        raise typer.Exit(code=1)

    # 提取基础文件名并去重排序
    base_names = set()
    for file_path in srt_files:
        # 转换为Path对象并获取相对于input_dir的路径
        file = Path(file_path)
        relative_path = file.relative_to(input_dir)
        file_name = relative_path.name

        # 移除扩展名
        base_name = re.sub(r'\.srt$', '', file_name, flags=re.IGNORECASE)

        # 移除各种语言后缀
        base_name = remove_language_suffix(base_name)
        base_names.add(base_name)

    # 自然排序基础文件名（EP2 在 EP10 之前）
    base_names = sorted(base_names, key=natural_sort_key)

    # 为每个基础名称找到对应的输入文件
    files_to_process = []
    for base_name in base_names:
        # 跳过已存在.ass文件的
        ass_file = input_dir / f"{base_name}.ass"
        if ass_file.exists():
            continue

        # 查找 SRT 文件
        candidate = input_dir / f"{base_name}.srt"
        if candidate.exists():
            files_to_process.append(candidate)
            print(f"📄 发现文件 [cyan]{candidate}[/cyan]")
        else:
            print(f"❌ 没有找到 [yellow]{base_name}[/yellow] 的 SRT 文件")

    if not files_to_process:
        print("[bold yellow]没有找到需要处理的新文件。[/bold yellow]")
        raise typer.Exit(code=0)

    # 应用数量限制
    if max_count > 0:
        files_to_process = files_to_process[:max_count]

    print(f"[bold green]开始批量翻译处理，共{len(files_to_process)}个文件...[/bold green]")
    if llm_model:
        print(f"使用LLM模型: [bold cyan]{llm_model}[/bold cyan]")

    return files_to_process
