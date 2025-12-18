import datetime
import json
import os
import time
import glob
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from mlx.core import bfloat16, float32
from rich import print
from rich.console import Console

# 抑制 macOS MallocStackLogging 警告（无害的系统消息）
os.environ.setdefault('MallocStackLogging', '0')
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from typing_extensions import Annotated

from ..logger import get_log_file_path, get_log_mode_info, setup_logger
from . import AlignedResult, AlignedSentence, AlignedToken, from_pretrained
from .utils import _find_cached_model, _check_network_connectivity, _storage_optimizer
from .model_cache import model_context, get_cache_info, clear_model_cache

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
    ] = True,
    chunk_duration: Annotated[
        float,
        typer.Option(
            help="长音频的分块时长（秒），0 禁用分块，-1 智能分块（推荐）。注意：当设为正数时，将使用固定分块而非 VAD 智能分块。"
        ),
    ] = -1,  # 智能分块：根据系统性能自动优化
    overlap_duration: Annotated[
        float, typer.Option(help="使用分块时的重叠时长（秒）")
    ] = 30,  # 优化：增加到30秒重叠，提高合并成功率
    use_vad: Annotated[
        bool,
        typer.Option(
            "--vad/--no-vad",
            help="使用 VAD 智能分块，在静音处分割音频（推荐）。仅在 chunk_duration 为负数时启用。"
        ),
    ] = True,
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
    
    # 显示日志文件路径信息
    log_mode, log_location = get_log_mode_info()
    log_path = get_log_file_path()
    print(f"📝 [dim]日志模式: {log_mode} ({log_location})[/dim]")
    print(f"📝 [dim]日志文件: {log_path}[/dim]")
    print()  # 空行分隔
    
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
        use_vad=use_vad,
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
    use_vad: bool,
    verbose: bool,
    fp32: bool
):
    """执行音频转录的核心逻辑"""
    # 设置日志记录器
    logger = setup_logger(__name__)

    # 不再在这里立即加载模型，而是延迟到实际转录时加载
    if verbose:
        print(f"准备使用模型: [bold cyan]{model}[/bold cyan]")
        print("🚀 [dim]模型将在开始转录时加载，支持批量处理优化[/dim]")

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

    # 使用批量处理模式的模型缓存管理
    batch_mode = total_files > 1
    
    with model_context(batch_mode=batch_mode):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("正在转录...", total=total_files)
            loaded_model = None  # 延迟初始化
            
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
                    # 懒加载模型（只在第一次需要时加载，后续复用缓存）
                    if loaded_model is None:
                        if verbose:
                            print(f"🤖 [bold blue]正在加载模型...[/bold blue] [cyan]{model}[/cyan]")

                        loaded_model, from_cache = from_pretrained(
                            model,
                            dtype=bfloat16 if not fp32 else float32,
                            show_progress=verbose,
                            use_cache=True,  # 启用缓存
                            return_cache_info=True  # 返回缓存信息
                        )

                        # 只有当模型不是从缓存加载时才显示加载完成信息
                        if verbose and not from_cache:
                            if batch_mode:
                                print("✅ [green]模型加载完成，批量处理模式已启用[/green]")
                            else:
                                print("✅ [green]模型加载完成[/green]")

                    # 记录转录开始时间
                    transcribe_start_time = time.time()

                    result: AlignedResult = loaded_model.transcribe(
                        audio_path,
                        dtype=bfloat16 if not fp32 else float32,
                        chunk_duration=chunk_duration if chunk_duration != 0 else None,
                        overlap_duration=overlap_duration,
                        use_vad=use_vad,
                        chunk_callback=lambda current, full: progress.update(
                            task, total=total_files * full, completed=full * i + current
                        ),
                    )

                    # 计算转录耗时
                    transcribe_elapsed = time.time() - transcribe_start_time

                    if verbose:
                        for sentence in result.sentences:
                            start, end, text = sentence.start, sentence.end, sentence.text
                            line = f"[blue][{format_timestamp(start)} --> {format_timestamp(end)}][/blue] {text.strip()}"
                            print(line)

                    # 统计字幕数量并显示时间统计
                    sentence_count = len(result.sentences)
                    logger.info(f"⏱️  转录耗时: {transcribe_elapsed:.1f}秒")
                    if not verbose:
                        print(f"✅ [bold green]转录完成[/bold green] (共 [cyan]{sentence_count}[/cyan] 条字幕) - 耗时: [cyan]{transcribe_elapsed:.1f}秒[/cyan]")

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
                        
                        # 对于SRT格式，检查内容是否为空
                        if fmt == "srt":
                            srt_content_trimmed = output_content.strip()
                            if not srt_content_trimmed:
                                print(f"[yellow]⚠️  未检测到语音内容，跳过保存 {fmt.upper()} 文件[/yellow]")
                                continue
                            
                            # 检查是否只包含空的时间戳条目（没有实际文本内容）
                            import re
                            # 移除序号和时间戳行，检查是否还有实际内容
                            lines = srt_content_trimmed.split('\n')
                            content_lines = []
                            for line in lines:
                                line = line.strip()
                                # 跳过序号行
                                if line.isdigit():
                                    continue
                                # 跳过时间戳行
                                if re.match(r'\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}', line):
                                    continue
                                # 跳过空行
                                if not line:
                                    continue
                                content_lines.append(line)
                            
                            if not content_lines:
                                print(f"[yellow]⚠️  转录结果为空，跳过保存 {fmt.upper()} 文件[/yellow]")
                                continue
                        
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
                    # 区分模型加载错误和转录错误
                    if loaded_model is None:
                        print(f"[bold red]加载模型 {model} 时出错:[/bold red] {e}")
                        # 模型加载失败，终止整个批次
                        raise typer.Exit(code=1)
                    else:
                        print(f"[bold red]转录文件 {audio_path} 时出错:[/bold red] {e}")
                        # 单个文件转录失败，继续处理其他文件

                progress.update(task, total=total_files, completed=i + 1)

        # 批量处理完成后，模型缓存将自动释放
        if verbose and batch_mode:
            print("🎯 [dim]批量处理完成，模型缓存已自动释放[/dim]")

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
    ] = True,
    chunk_duration: Annotated[
        float,
        typer.Option(
            help="长音频的分块时长（秒），0 禁用分块，-1 智能分块（推荐）。注意：当设为正数时，将使用固定分块而非 VAD 智能分块。"
        ),
    ] = -1,  # 智能分块：根据系统性能自动优化
    overlap_duration: Annotated[
        float, typer.Option(help="使用分块时的重叠时长（秒）")
    ] = 30,  # 优化：增加到30秒重叠，提高合并成功率
    use_vad: Annotated[
        bool,
        typer.Option(
            "--vad/--no-vad",
            help="使用 VAD 智能分块，在静音处分割音频（推荐）。仅在 chunk_duration 为负数时启用。"
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="打印详细的处理和调试信息"),
    ] = False,
    fp32: Annotated[
        bool, typer.Option("--fp32/--bf16", help="使用 FP32 精度")
    ] = False,
    version: Annotated[
        bool, typer.Option("--version", help="显示版本信息并退出")
    ] = False,
):
    """使用 Parakeet MLX 模型转录音频文件。"""
    # 处理版本信息请求
    if version:
        from ..version_utils import get_simple_version_info
        print(get_simple_version_info())
        raise typer.Exit()
    
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
    
    # 如果没有提供音频文件，尝试扫描当前目录
    if not audio_files:
        print("[dim]未指定输入文件，扫描当前目录...[/dim]")
        try:
            audio_files = _scan_media_files(Path.cwd())
        except typer.Exit as e:
            # 如果扫描也没找到文件，会抛出Exit异常
            raise e

    # 如果仍然没有音频文件（理论上_scan_media_files会处理这种情况，但为了保险起见）
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
        use_vad=use_vad,
        verbose=verbose,
        fp32=fp32
    )

def _scan_media_files(input_dir: Path) -> List[Path]:
    """扫描目录下的媒体文件，排除已有字幕的文件"""
    # 媒体文件扩展名
    MEDIA_EXTENSIONS = [
        # 音频格式
        "*.mp3", "*.m4a", "*.wav", "*.flac", "*.aac",
        "*.ogg", "*.wma", "*.aiff", "*.opus",
        # 视频格式
        "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm",
        "*.flv", "*.wmv", "*.m4v", "*.mpeg", "*.mpg",
        "*.3gp", "*.ts"
    ]

    # 确保input_dir是绝对路径
    input_dir = input_dir.resolve()

    # 查找所有媒体文件
    media_files = []
    for pattern in MEDIA_EXTENSIONS:
        media_files.extend(glob.glob(str(input_dir / pattern)))
    
    if not media_files:
        print(f"[bold red]{input_dir} 目录中没有找到需要处理的媒体文件。[/bold red]")
        raise typer.Exit(code=1)
    
    # 过滤掉已有字幕的文件
    files_to_process = []
    
    # 提取基础文件名并去重排序
    # 这里我们需要保留原始文件路径，所以不能像translate那样只存basename
    # 我们用basename来检查是否有对应的subtitle文件
    
    # 排序文件列表
    media_files.sort()
    
    for file_path in media_files:
        file = Path(file_path)
        
        # 检查是否存在对应的字幕文件 (.srt 或 .ass)
        # 注意：这里简化处理，只检查同名不同后缀的情况
        # translate命令中的逻辑比较复杂，这里先做最基础的检查
        has_subtitle = False
        for sub_ext in ['.srt', '.ass', '.vtt']:
            if (file.parent / (file.stem + sub_ext)).exists():
                has_subtitle = True
                break
        
        if not has_subtitle:
            files_to_process.append(file)
    
    if not files_to_process:
        print("[bold yellow]目录下所有媒体文件均已有字幕，没有需要处理的新文件。[/bold yellow]")
        raise typer.Exit(code=0)
        
    print(f"[bold green]发现 {len(files_to_process)} 个待转录文件：[/bold green]")
    for f in files_to_process[:5]:
        print(f"  - {f.name}")
    if len(files_to_process) > 5:
        print(f"  ... 等共 {len(files_to_process)} 个文件")
    print()
        
    return files_to_process


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
        console.print("💡 支持的操作: list, info, download, clean, cache")
        console.print("\n📖 使用示例:")
        console.print("   transcribe model list                                    # 列出已缓存转录模型")
        console.print("   transcribe model info                                    # 显示默认转录模型信息")
        console.print("   transcribe model info mlx-community/parakeet-tdt-0.6b-v2  # 显示指定转录模型信息")
        console.print("   transcribe model download                                      # 预下载默认转录模型")
        console.print("   transcribe model download mlx-community/parakeet-tdt-0.6b-v2  # 预下载指定转录模型")
        console.print("   transcribe model clean                                   # 清理缓存")
        console.print("   transcribe model cache status                           # 查看缓存状态")
        console.print("   transcribe model cache clear                            # 清理内存缓存")


@app.command("version") 
def version():
    """显示版本信息"""
    from rich.console import Console
    from ..version_utils import display_version_info
    
    console = Console()
    display_version_info(console)


@app.command("cache")
def cache_cmd(
    ctx: typer.Context,
    action: str = typer.Argument(..., help="缓存操作: status(查看状态), clear(清理内存缓存), optimize(清理存储优化缓存)")
):
    """模型缓存管理命令"""
    
    if action == "status":
        """显示缓存状态信息"""
        try:
            # 获取内存缓存信息
            cache_info = get_cache_info()
            
            # 获取存储优化缓存信息
            storage_stats = _storage_optimizer.get_cache_stats()
            
            # 创建状态表格
            table = Table(title="🧠 模型缓存状态")
            table.add_column("缓存类型", style="cyan")
            table.add_column("状态", style="green")
            table.add_column("详细信息", style="dim")
            
            # 内存缓存状态
            if cache_info["status"] == "cached":
                table.add_row(
                    "内存缓存",
                    "✅ 已缓存",
                    f"模型: {cache_info['model_id']}, 类型: {cache_info['dtype']}, 访问: {cache_info['access_count']}次"
                )
                if cache_info.get("batch_mode", False):
                    table.add_row("", "🔄 批量模式", f"引用计数: {cache_info.get('batch_ref_count', 0)}")
            else:
                table.add_row("内存缓存", "❌ 空闲", "无模型缓存")
            
            # 存储优化缓存状态
            if storage_stats["cached_models"] > 0:
                table.add_row(
                    "存储优化缓存",
                    "✅ 可用",
                    f"{storage_stats['cached_models']} 个模型, {storage_stats['total_size_mb']:.1f} MB"
                )
            else:
                table.add_row("存储优化缓存", "❌ 空白", "无优化缓存")
            
            console.print(table)
            
            # 显示缓存位置信息
            if storage_stats.get("cache_dir"):
                console.print(f"\n📍 存储位置: [dim]{storage_stats['cache_dir']}[/dim]")
                
        except Exception as e:
            console.print(f"[red]❌ 获取缓存状态失败: {str(e)}[/red]")
    
    elif action == "clear":
        """清理内存缓存"""
        try:
            cache_info = get_cache_info()
            
            if cache_info["status"] == "cached":
                console.print(f"⚠️  [yellow]即将清理内存中的模型缓存[/yellow]")
                console.print(f"模型: [cyan]{cache_info['model_id']}[/cyan]")
                
                confirm = typer.confirm("确定要清理内存缓存吗？")
                if not confirm:
                    console.print("❌ 取消清理操作")
                    return
                
                clear_model_cache()
                console.print("✅ [green]内存缓存已清理[/green]")
                console.print("💡 [dim]下次使用时将重新从存储优化缓存或原始文件加载[/dim]")
            else:
                console.print("[yellow]📂 内存缓存为空，无需清理[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ 清理内存缓存失败: {str(e)}[/red]")
    
    elif action == "optimize":
        """清理存储优化缓存"""
        try:
            storage_stats = _storage_optimizer.get_cache_stats()
            
            if storage_stats["cached_models"] > 0:
                console.print(f"⚠️  [yellow]即将清理存储优化缓存[/yellow]")
                console.print(f"缓存模型: {storage_stats['cached_models']} 个")
                console.print(f"占用空间: {storage_stats['total_size_mb']:.1f} MB")
                
                confirm = typer.confirm("确定要清理存储优化缓存吗？")
                if not confirm:
                    console.print("❌ 取消清理操作")
                    return
                
                _storage_optimizer.clear_all_optimized_cache()
                console.print("✅ [green]存储优化缓存已清理[/green]")
                console.print("💡 [dim]下次使用时将从原始文件重新构建优化缓存[/dim]")
            else:
                console.print("[yellow]📂 存储优化缓存为空，无需清理[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ 清理存储优化缓存失败: {str(e)}[/red]")
    
    else:
        console.print(f"[red]❌ 未知操作: {action}[/red]")
        console.print("💡 支持的操作: status, clear, optimize")
        console.print("\n📖 使用示例:")
        console.print("   transcribe cache status                           # 查看缓存状态")
        console.print("   transcribe cache clear                            # 清理内存缓存")
        console.print("   transcribe cache optimize                         # 清理存储优化缓存")


if __name__ == "__main__":
    app()
