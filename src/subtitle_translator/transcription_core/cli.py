import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from mlx.core import bfloat16, float32
from rich import print
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from typing_extensions import Annotated

from . import AlignedResult, AlignedSentence, AlignedToken, from_pretrained
from .utils import _find_cached_model, _check_network_connectivity

# 默认转录模型
DEFAULT_TRANSCRIPTION_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"

# 初始化控制台
console = Console()

# helpers
def format_timestamp(
    seconds: float, always_include_hours: bool = True, decimal_marker: str = ","
) -> str:
    assert seconds >= 0
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def to_txt(result: AlignedResult) -> str:
    """Format transcription result as plain text."""
    return result.text.strip()


def to_srt(result: AlignedResult, timestamps: bool = False) -> str:
    """
    Format transcription result as an SRT file.
    """
    srt_content = []
    entry_index = 1
    if timestamps:
        words = result.words
        for word in words:
            if not word.text.strip():
                continue
            start_time = format_timestamp(word.start, decimal_marker=",")
            end_time = format_timestamp(word.end, decimal_marker=",")
            text = word.text.strip()

            srt_content.append(f"{entry_index}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")
            entry_index += 1
    else:
        for sentence in result.sentences:
            start_time = format_timestamp(sentence.start, decimal_marker=",")
            end_time = format_timestamp(sentence.end, decimal_marker=",")
            text = sentence.text.strip()

            srt_content.append(f"{entry_index}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")
            entry_index += 1

    return "\n".join(srt_content)


def to_vtt(result: AlignedResult, timestamps: bool = False) -> str:
    """
    Format transcription result as a VTT file.
    """
    vtt_content = ["WEBVTT", ""]
    if timestamps:
        words = result.words
        for word in words:
            if not word.text.strip():
                continue
            start_time = format_timestamp(word.start, decimal_marker=".")
            end_time = format_timestamp(word.end, decimal_marker=".")
            text = word.text.strip()

            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")
    else:
        for sentence in result.sentences:
            start_time = format_timestamp(sentence.start, decimal_marker=".")
            end_time = format_timestamp(sentence.end, decimal_marker=".")
            text_line = sentence.text.strip()

            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text_line)
            vtt_content.append("")

    return "\n".join(vtt_content)


def _aligned_token_to_dict(token: AlignedToken) -> Dict[str, Any]:
    return {
        "text": token.text,
        "start": round(token.start, 3),
        "end": round(token.end, 3),
        "duration": round(token.duration, 3),
    }


def _aligned_sentence_to_dict(sentence: AlignedSentence) -> Dict[str, Any]:
    return {
        "text": sentence.text,
        "start": round(sentence.start, 3),
        "end": round(sentence.end, 3),
        "duration": round(sentence.duration, 3),
        "tokens": [_aligned_token_to_dict(token) for token in sentence.tokens],
    }


def to_json(result: AlignedResult) -> str:
    output_dict = {
        "text": result.text,
        "sentences": [
            _aligned_sentence_to_dict(sentence) for sentence in result.sentences
        ],
    }
    return json.dumps(output_dict, indent=2, ensure_ascii=False)


app = typer.Typer(
    help="使用 Parakeet MLX 模型进行音频转录的命令行工具",
    epilog="💡 首次使用前可以运行: transcribe model download <model_id> 来预下载模型"
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    audios: Annotated[
        Optional[List[Path]],
        typer.Argument(
            help="要转录的音频文件",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    model: Annotated[
        str, typer.Option(help="要使用的 Hugging Face 模型仓库")
    ] = "mlx-community/parakeet-tdt-0.6b-v2",
    output_dir: Annotated[
        Path, typer.Option(help="保存转录结果的目录")
    ] = Path("."),
    output_format: Annotated[
        str, typer.Option(help="输出文件格式 (txt, srt, vtt, json, all)")
    ] = "srt",
    output_template: Annotated[
        str,
        typer.Option(
            help="输出文件名模板，例如 '{filename}_{date}_{index}'"
        ),
    ] = "{filename}",
    timestamps: Annotated[
        bool,
        typer.Option(help="在 srt/vtt 格式中输出词级时间戳"),
    ] = False,
    chunk_duration: Annotated[
        float,
        typer.Option(
            help="长音频的分块时长（秒），0 禁用分块。"
        ),
    ] = 60 * 2,
    overlap_duration: Annotated[
        float, typer.Option(help="使用分块时的重叠时长（秒）")
    ] = 15,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="打印详细的处理和调试信息"),
    ] = False,
    fp32: Annotated[
        bool, typer.Option("--fp32/--bf16", help="使用 FP32 精度")
    ] = False,
):
    """
    使用 Parakeet MLX 模型转录音频文件。
    """
    # 如果调用了子命令，就不执行主逻辑
    if ctx.invoked_subcommand is not None:
        return
    
    # 如果没有提供音频文件，显示帮助信息
    if not audios:
        print("[bold red]错误: 请提供要转录的音频文件[/bold red]")
        print("\n💡 [bold blue]使用示例:[/bold blue]")
        print("   transcribe audio.mp3 audio2.wav                    # 转录多个文件")
        print("   transcribe *.mp3 --output-format all              # 转录所有mp3文件为所有格式")
        print("   transcribe audio.wav --model other-model          # 使用指定模型")
        print("   transcribe model list                             # 查看已缓存的模型")
        raise typer.Exit(code=1)
    
    _transcribe_files(
        audios=audios,
        model=model,
        output_dir=output_dir,
        output_format=output_format,
        output_template=output_template,
        timestamps=timestamps,
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
        verbose=verbose,
        fp32=fp32
    )


def _transcribe_files(
    audios: List[Path],
    model: str,
    output_dir: Path,
    output_format: str,
    output_template: str,
    timestamps: bool,
    chunk_duration: float,
    overlap_duration: float,
    verbose: bool,
    fp32: bool
):
    """执行音频转录的核心逻辑"""
    if verbose:
        print(f"正在加载模型: [bold cyan]{model}[/bold cyan]...")

    try:
        # 使用增强的模型加载体验
        loaded_model = from_pretrained(
            model, 
            dtype=bfloat16 if not fp32 else float32,
            show_progress=verbose
        )
        if verbose:
            print("[green]模型加载成功。[/green]")
    except Exception as e:
        print(f"[bold red]加载模型 {model} 时出错:[/bold red] {e}")
        raise typer.Exit(code=1)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[bold red]创建输出目录 {output_dir} 时出错:[/bold red] {e}")
        raise typer.Exit(code=1)

    if verbose:
        print(f"输出目录: [bold cyan]{output_dir.resolve()}[/bold cyan]")
        print(f"输出格式: [bold cyan]{output_format}[/bold cyan]")
        if output_format in ["srt", "vtt", "all"] and timestamps:
            print("时间戳: [bold cyan]已启用[/bold cyan]")

    formatters = {
        "txt": to_txt,
        "srt": lambda r: to_srt(r, timestamps=timestamps),
        "vtt": lambda r: to_vtt(r, timestamps=timestamps),
        "json": to_json,
    }

    formats_to_generate = []
    if output_format == "all":
        formats_to_generate = list(formatters.keys())
    elif output_format in formatters:
        formats_to_generate = [output_format]
    else:
        print(
            f"[bold red]错误: 无效的输出格式 '{output_format}'。请从 {list(formatters.keys()) + ['all']} 中选择。[/bold red]"
        )
        raise typer.Exit(code=1)

    total_files = len(audios)
    if verbose:
        print(f"正在转录 {total_files} 个文件...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("正在转录...", total=total_files)

        for i, audio_path in enumerate(audios):
            if verbose:
                print(
                    f"\n正在处理文件 {i + 1}/{total_files}: [bold cyan]{audio_path.name}[/bold cyan]"
                )
            else:
                progress.update(
                    task, description=f"正在处理 [cyan]{audio_path.name}[/cyan]..."
                )

            try:
                result: AlignedResult = loaded_model.transcribe(
                    audio_path,
                    dtype=bfloat16 if not fp32 else float32,
                    chunk_duration=chunk_duration if chunk_duration != 0 else None,
                    overlap_duration=overlap_duration,
                    chunk_callback=lambda current, full: progress.update(
                        task, total=total_files * full, completed=full * i + current
                    ),
                )

                if verbose:
                    for sentence in result.sentences:
                        start, end, text = sentence.start, sentence.end, sentence.text
                        line = f"[blue][{format_timestamp(start)} --> {format_timestamp(end)}][/blue] {text.strip()}"
                        print(line)

                base_filename = audio_path.stem
                template_vars = {
                    "filename": base_filename,
                    "date": datetime.datetime.now().strftime("%Y%m%d"),
                    "index": str(i + 1),
                }

                output_basename = output_template.format(**template_vars)

                for fmt in formats_to_generate:
                    formatter = formatters[fmt]
                    output_content = formatter(result)
                    output_filename = f"{output_basename}.{fmt}"
                    output_filepath = output_dir / output_filename

                    try:
                        with open(output_filepath, "w", encoding="utf-8") as f:
                            f.write(output_content)
                        if verbose:
                            print(
                                f"[green]已保存 {fmt.upper()}:[/green] {output_filepath.absolute()}"
                            )
                    except Exception as e:
                        print(
                            f"[bold red]写入输出文件 {output_filepath} 时出错:[/bold red] {e}"
                        )

            except Exception as e:
                print(f"[bold red]转录文件 {audio_path} 时出错:[/bold red] {e}")

            progress.update(task, total=total_files, completed=i + 1)

    print(
        f"\n[bold green]转录完成。[/bold green] 输出已保存在 '{output_dir.resolve()}'"
    )


app = typer.Typer(
    help="使用 Parakeet MLX 模型进行音频转录的命令行工具",
    epilog="💡 首次使用前可以运行: transcribe model download <model_id> 来预下载模型"
)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    input_files: Annotated[
        Optional[str],
        typer.Option(
            "--input-files", "-i",
            help="要转录的音频文件（用逗号分隔多个文件），如不指定则使用位置参数"
        )
    ] = None,
    model: Annotated[
        str, typer.Option(help="要使用的 Hugging Face 模型仓库")
    ] = "mlx-community/parakeet-tdt-0.6b-v2",
    output_dir: Annotated[
        Path, typer.Option(help="保存转录结果的目录")
    ] = Path("."),
    output_format: Annotated[
        str, typer.Option(help="输出文件格式 (txt, srt, vtt, json, all)")
    ] = "srt",
    output_template: Annotated[
        str,
        typer.Option(
            help="输出文件名模板，例如 '{filename}_{date}_{index}'"
        ),
    ] = "{filename}",
    timestamps: Annotated[
        bool,
        typer.Option(help="在 srt/vtt 格式中输出词级时间戳"),
    ] = False,
    chunk_duration: Annotated[
        float,
        typer.Option(
            help="长音频的分块时长（秒），0 禁用分块。"
        ),
    ] = 60 * 2,
    overlap_duration: Annotated[
        float, typer.Option(help="使用分块时的重叠时长（秒）")
    ] = 15,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="打印详细的处理和调试信息"),
    ] = False,
    fp32: Annotated[
        bool, typer.Option("--fp32/--bf16", help="使用 FP32 精度")
    ] = False,
):
    """使用 Parakeet MLX 模型转录音频文件。"""
    # 如果调用了子命令，就不执行主逻辑
    if ctx.invoked_subcommand is not None:
        return
    
    # 获取要处理的文件列表
    audio_files = []
    
    # 处理 -i 参数指定的文件
    if input_files:
        for file_path_str in input_files.split(","):
            file_path = Path(file_path_str.strip())
            if not file_path.exists():
                print(f"[bold red]错误: 文件不存在: {file_path}[/bold red]")
                raise typer.Exit(code=1)
            audio_files.append(file_path)
    else:
        # 处理位置参数（剩余参数）
        remaining_args = ctx.args
        if remaining_args:
            for file_path_str in remaining_args:
                file_path = Path(file_path_str)
                if not file_path.exists():
                    print(f"[bold red]错误: 文件不存在: {file_path}[/bold red]")
                    raise typer.Exit(code=1)
                audio_files.append(file_path)
    
    # 如果没有提供音频文件，显示帮助信息
    if not audio_files:
        print("[bold red]错误: 请提供要转录的音频文件[/bold red]")
        print("\n💡 [bold blue]使用示例:[/bold blue]")
        print("   transcribe audio.mp3 audio2.wav                    # 直接转录多个文件")
        print("   transcribe -i audio.mp3,audio2.wav                 # 使用 -i 参数")
        print("   transcribe *.mp3 --output-format all              # 转录所有mp3文件")
        print("   transcribe model list                             # 查看已缓存的模型")
        raise typer.Exit(code=1)
    
    # 调用转录功能
    _transcribe_files(
        audios=audio_files,
        model=model,
        output_dir=output_dir,
        output_format=output_format,
        output_template=output_template,
        timestamps=timestamps,
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
        verbose=verbose,
        fp32=fp32
    )


@app.command("model")
def model_cmd(
    ctx: typer.Context,
    action: str = typer.Argument(..., help="要执行的操作: list(列出已缓存模型), info(显示模型信息), download(预下载模型), clean(清理缓存)"),
    model_id: Optional[str] = typer.Argument(None, help=f"模型ID (download和info操作默认: {DEFAULT_TRANSCRIPTION_MODEL})")
):
    """转录模型管理命令"""
    import os
    import shutil
    
    if action == "list":
        """列出已缓存的转录模型"""
        try:
            # 获取缓存目录
            cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or Path.home() / ".cache" / "huggingface"
            cache_dir = Path(cache_dir) / "hub"
            
            if not cache_dir.exists():
                console.print("[yellow]📂 还没有缓存任何转录模型[/yellow]")
                return
            
            # 查找模型缓存目录
            model_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith("models--")]
            
            if not model_dirs:
                console.print("[yellow]📂 还没有缓存任何转录模型[/yellow]")
                return
            
            # 创建表格显示模型信息
            table = Table(title="🎤 已缓存的转录模型列表")
            table.add_column("模型ID", style="cyan")
            table.add_column("缓存大小", style="green")
            table.add_column("最后修改时间", style="dim")
            
            for model_dir in sorted(model_dirs):
                # 解析模型ID
                model_id = model_dir.name.replace("models--", "").replace("--", "/")
                
                # 计算目录大小
                total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                
                # 获取最后修改时间
                import datetime
                mtime = datetime.datetime.fromtimestamp(model_dir.stat().st_mtime)
                
                table.add_row(
                    model_id,
                    f"{size_mb:.1f} MB",
                    mtime.strftime("%Y-%m-%d %H:%M")
                )
            
            console.print(table)
            console.print(f"\n📍 缓存位置: [dim]{cache_dir}[/dim]")
            
        except Exception as e:
            console.print(f"[red]❌ 获取转录模型列表失败: {str(e)}[/red]")
    
    elif action == "info":
        """显示指定转录模型的详细信息"""
        # 如果没有指定模型ID，使用默认模型
        if not model_id:
            model_id = DEFAULT_TRANSCRIPTION_MODEL
            console.print(f"[dim]使用默认转录模型: {model_id}[/dim]")
        
        try:
            # 尝试查找本地缓存
            try:
                config_path, weight_path = _find_cached_model(model_id)
                console.print(f"✅ [green]转录模型已缓存[/green]: [bold]{model_id}[/bold]")
                console.print(f"📄 配置文件: [dim]{config_path}[/dim]")
                console.print(f"⚖️  权重文件: [dim]{weight_path}[/dim]")
                
                # 显示文件大小
                config_size = Path(config_path).stat().st_size / 1024
                weight_size = Path(weight_path).stat().st_size / (1024 * 1024)
                console.print(f"📊 大小: 配置 {config_size:.1f} KB, 权重 {weight_size:.1f} MB")
                
            except FileNotFoundError:
                console.print(f"[yellow]⚠️  转录模型未缓存[/yellow]: [bold]{model_id}[/bold]")
                console.print("💡 你可以使用 'transcribe model download' 命令预下载模型")
                
                # 检查网络连接
                if _check_network_connectivity():
                    console.print("🌐 网络连接正常，模型将在首次使用时自动下载")
                else:
                    console.print("[red]🌐 网络连接异常，无法下载模型[/red]")
                    
        except Exception as e:
            console.print(f"[red]❌ 获取转录模型信息失败: {str(e)}[/red]")
    
    elif action == "download":
        """预下载指定转录模型"""
        # 如果没有指定模型ID，使用默认模型
        if not model_id:
            model_id = DEFAULT_TRANSCRIPTION_MODEL
            console.print(f"[dim]使用默认转录模型: {model_id}[/dim]")
        
        try:
            console.print(f"🚀 开始预下载转录模型: [bold]{model_id}[/bold]")
            
            # 检查是否已经缓存
            try:
                _find_cached_model(model_id)
                console.print(f"✅ [green]转录模型已存在于本地缓存[/green]")
                return
            except FileNotFoundError:
                pass
            
            # 下载模型
            model = from_pretrained(model_id, show_progress=True)
            console.print(f"\n🎉 [bold green]转录模型预下载完成![/bold green]")
            console.print(f"📍 模型已保存到本地缓存，后续使用时将直接加载")
            
        except Exception as e:
            console.print(f"[red]❌ 转录模型下载失败: {str(e)}[/red]")
    
    elif action == "clean":
        """清理转录模型缓存"""
        try:
            # 获取缓存目录
            cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or Path.home() / ".cache" / "huggingface"
            cache_dir = Path(cache_dir) / "hub"
            
            if not cache_dir.exists():
                console.print("[yellow]📂 缓存目录不存在，无需清理[/yellow]")
                return
            
            # 计算缓存大小
            total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            # 询问确认
            if size_mb > 0:
                console.print(f"⚠️  [yellow]即将清理 {size_mb:.1f} MB 的转录模型缓存[/yellow]")
                console.print(f"📍 缓存位置: [dim]{cache_dir}[/dim]")
                
                confirm = typer.confirm("确定要清理所有转录模型缓存吗？")
                if not confirm:
                    console.print("❌ 取消清理操作")
                    return
                
                # 清理缓存
                shutil.rmtree(cache_dir)
                console.print("✅ [green]转录模型缓存清理完成[/green]")
            else:
                console.print("[yellow]📂 缓存目录为空，无需清理[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ 清理缓存失败: {str(e)}[/red]")
    
    else:
        console.print(f"[red]❌ 未知操作: {action}[/red]")
        console.print("💡 支持的操作: list, info, download, clean")
        console.print("\n📖 使用示例:")
        console.print("   transcribe model list                                    # 列出已缓存转录模型")
        console.print("   transcribe model info                                    # 显示默认转录模型信息")
        console.print("   transcribe model info mlx-community/parakeet-tdt-0.6b-v2  # 显示指定转录模型信息")
        console.print("   transcribe model download                                      # 预下载默认转录模型")
        console.print("   transcribe model download mlx-community/parakeet-tdt-0.6b-v2  # 预下载指定转录模型")
        console.print("   transcribe model clean                                   # 清理缓存")


if __name__ == "__main__":
    app()