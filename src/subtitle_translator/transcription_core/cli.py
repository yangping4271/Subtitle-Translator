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
from .model_cache import model_context

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
    help="使用 Parakeet MLX 模型进行音频转录的命令行工具"
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

                        loaded_model = from_pretrained(
                            model,
                            dtype=bfloat16 if not fp32 else float32,
                            use_cache=True
                        )

                        if verbose:
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


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    audios: Annotated[
        Optional[List[str]],
        typer.Argument(help="要转录的音频文件")
    ] = None,
    input_files: Annotated[
        Optional[str],
        typer.Option(
            "--input-files", "-i",
            help="要转录的音频文件（用逗号分隔多个文件），如不指定则使用位置参数"
        )
    ] = None,
    model: Annotated[
        Optional[str], typer.Option(help="转录模型路径（可选，默认使用配置文件中的 TRANSCRIPTION_MODEL_PATH 或 HF 缓存）")
    ] = None,
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

    # 处理位置参数
    if audios:
        for file_path_str in audios:
            file_path = Path(file_path_str)
            if not file_path.exists():
                print(f"[bold red]错误: 文件不存在: {file_path}[/bold red]")
                raise typer.Exit(code=1)
            audio_files.append(file_path)

    # 处理 -i 参数指定的文件
    if input_files:
        for file_path_str in input_files.split(","):
            file_path = Path(file_path_str.strip())
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


@app.command("version")
def version():
    """显示版本信息"""
    from rich.console import Console
    from ..version_utils import display_version_info

    console = Console()
    display_version_info(console)


def cli_main():
    """CLI入口点包装器，捕获所有未处理异常，避免输出 traceback"""
    try:
        app()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[bold red]❌ 发生错误:[/bold red] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    cli_main()
