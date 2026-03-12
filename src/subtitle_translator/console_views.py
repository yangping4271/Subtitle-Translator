"""
控制台视图模块 - 负责格式化输出和用户界面展示
"""
from pathlib import Path
from typing import List, Optional

from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .file_discovery import get_file_type_info, format_file_size


def show_api_config(base_url: str, api_key: str) -> None:
    """显示 API 配置信息"""
    print("[bold blue]🌐 API 配置:[/bold blue]")
    print(f"   端点: [cyan]{base_url}[/cyan]")
    masked_key = f"{api_key[:6]}{'*' * 8}{api_key[-6:]}" if len(api_key) > 12 else '*' * len(api_key)
    print(f"   密钥: [cyan]{masked_key}[/cyan]" if api_key else "   密钥: [red]未设置[/red]")


def show_model_config(split_model: str, translation_model: str) -> None:
    """显示模型配置信息"""
    print("[bold blue]🤖 模型配置:[/bold blue]")
    print(f"   断句: [cyan]{split_model}[/cyan]")
    print(f"   翻译: [cyan]{translation_model}[/cyan]")


def show_time_stats(stages: dict, total_time: float) -> None:
    """格式化显示时间统计"""
    print("[bold blue]⏱️  耗时统计:[/bold blue]")
    for stage_name, elapsed_time in stages.items():
        if elapsed_time > 0 and stage_name != "⚡ 并行预处理":
            percentage = (elapsed_time / total_time) * 100
            print(f"   {stage_name}: [cyan]{elapsed_time:.1f}s[/cyan] ([dim]{percentage:.0f}%[/dim])")
    print(f"   [bold]总计: [cyan]{total_time:.1f}s[/cyan][/bold]")


def show_dry_run_summary(files_to_process: List[Path], target_lang: str, output_dir: Path,
                         llm_model: Optional[str], input_dir: Path):
    """显示预览模式的文件处理信息"""
    console = Console()

    # 标题
    console.print("\n[bold blue]🔍 预览模式 - 将要处理的文件信息[/bold blue]\n")

    # 基本信息
    info_table = Table(show_header=False, box=box.ROUNDED, expand=False)
    info_table.add_column("项目", style="cyan", width=15)
    info_table.add_column("值", style="white")

    info_table.add_row("📁 输入目录", str(input_dir))
    info_table.add_row("📂 输出目录", str(output_dir))
    info_table.add_row("🎯 目标语言", target_lang)

    # 显示模型配置
    if llm_model:
        info_table.add_row("🤖 LLM模型", llm_model)

    console.print(info_table)
    console.print()

    # 文件列表
    if files_to_process:
        file_table = Table(title="📄 发现的文件列表", box=box.ROUNDED)
        file_table.add_column("序号", style="cyan", width=6, justify="right")
        file_table.add_column("文件名", style="white")
        file_table.add_column("类型", style="yellow")
        file_table.add_column("大小", style="green", justify="right")
        file_table.add_column("处理方式", style="magenta")

        for idx, file_path in enumerate(files_to_process, 1):
            file_name = file_path.name
            file_ext = file_path.suffix.lower()

            # 确定文件类型和处理方式
            file_type, process_type = get_file_type_info(file_ext)

            # 获取文件大小
            size_str = format_file_size(file_path)

            file_table.add_row(str(idx), file_name, file_type, size_str, process_type)

        console.print(file_table)

        # 统计信息
        total_size = sum(f.stat().st_size for f in files_to_process if f.exists())
        srt_count = len(files_to_process)  # 现在只有 SRT 文件

        summary = f"""
[bold]📊 处理统计:[/bold]
• 总文件数: {len(files_to_process)} 个
• 字幕文件: {srt_count} 个 (直接翻译)
• 总大小: {total_size / (1024 * 1024):.1f} MB
        """
        console.print(Panel(summary.strip(), title="[bold green]处理概览[/bold green]", border_style="green"))

    else:
        console.print("[bold yellow]⚠️  没有发现可处理的文件[/bold yellow]")

    # 提示信息
    tip_panel = Panel(
        "[bold cyan]💡 提示:[/bold cyan]\n"
        "• 移除 [bold magenta]--dry-run[/bold magenta] 参数以开始实际处理\n"
        "• 使用 [bold magenta]--count N[/bold magenta] 限制处理文件数量\n"
        "• 使用 [bold magenta]--output-dir[/bold magenta] 指定输出目录",
        title="[bold]操作指南[/bold]",
        border_style="cyan"
    )
    console.print("\n", tip_panel)


def show_results(count: int, generated_ass_files: List[Path], output_dir: Path, is_batch_mode: bool):
    """显示处理结果"""
    from .logger import setup_logger
    logger = setup_logger(__name__)

    print()
    if is_batch_mode:
        logger.info("🎉 批量处理完成！")
        logger.info(f"总计处理文件数: {count}")
        print(f"🎉 [bold green]批量处理完成！[/bold green] (处理 [cyan]{count}[/cyan] 个文件)")
    else:
        logger.info("🎉 处理完成！")
        logger.info(f"总计处理文件数: {count}")
        print(f"🎉 [bold green]处理完成！[/bold green] (处理 [cyan]{count}[/cyan] 个文件)")

    # 只显示本次生成的ASS文件统计
    if count > 0:
        if generated_ass_files:
            logger.info("本次生成的ASS文件：")
            for f in generated_ass_files:
                logger.info(f"  {f.name}")
            print(f"📺 [bold green]已生成 {len(generated_ass_files)} 个双语ASS文件[/bold green]")

        logger.info("处理完毕！")
