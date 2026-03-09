"""
主命令行接口模块 - 简洁清晰的CLI入口
"""
import glob
import os
import re
import typer
from pathlib import Path
from typing import Optional, List, Tuple
from typing_extensions import Annotated

from rich import print

from .env_setup import setup_environment
from .logger import setup_logger

DEFAULT_TRANSCRIPTION_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"

AUDIO_EXTENSIONS = ['.mp3', '.m4a', '.wav', '.flac', '.aac', '.ogg', '.wma', '.aiff', '.opus']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.mpeg', '.mpg', '.3gp', '.ts']
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS

logger = setup_logger(__name__)


app = typer.Typer(
    help="一个集成了语音转录、字幕翻译和格式转换的命令行工具"
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_file: Optional[Path] = typer.Option(None, "--input-file", "-i", help="要处理的单个文件路径，如不指定则批量处理当前目录或指定目录。", exists=True, file_okay=True, dir_okay=False, readable=True),
    input_dir: Optional[Path] = typer.Option(None, "--input-dir", help="批量处理时指定输入目录，不指定则使用当前目录。", exists=True, file_okay=False, dir_okay=True, readable=True),
    max_count: int = typer.Option(-1, "--count", "-n", help="最大处理文件数量，-1表示处理所有文件。"),
    target_lang: str = typer.Option("zh", "--target-lang", "-t", help="目标翻译语言。支持：zh/zh-cn(简中), zh-tw(繁中), ja(日), ko(韩), fr(法), de(德), es(西), pt(葡), it(意), ru(俄), ar(阿), th(泰), vi(越)等。"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="输出文件的目录，默认为当前目录。"),
    model: str = typer.Option(DEFAULT_TRANSCRIPTION_MODEL, "--model", help="用于转录的 Parakeet MLX 模型。"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", "-m", help="覆盖所有模型（优先级低于独立参数）"),
    split_model: Optional[str] = typer.Option(None, "--split-model", help="断句模型"),
    translation_model: Optional[str] = typer.Option(None, "--translation-model", help="翻译模型"),
    preserve_intermediate: bool = typer.Option(False, "--preserve-intermediate", "-p", help="保留中间的英文和目标语言SRT文件，便于进一步处理或调试。"),
    dry_run: bool = typer.Option(False, "--dry-run", help="预览模式，只显示将要处理的文件信息而不实际执行翻译。"),
    transcribe: bool = typer.Option(False, "--transcribe", help="当找不到字幕文件时，是否允许进行语音转录。"),
    version: bool = typer.Option(False, "--version", help="显示版本信息并退出。"),
):
    """字幕翻译工具主命令"""
    if version:
        from .version_utils import get_simple_version_info
        print(get_simple_version_info())
        raise typer.Exit()

    if ctx.invoked_subcommand is not None:
        return

    setup_environment()

    try:
        _validate_target_language(target_lang)
    except ValueError as e:
        logger.error(f"❌ 命令行参数错误 - 目标语言: {str(e)}")
        print(f"[bold red]❌ 目标语言参数错误![/bold red]")
        print(str(e))
        print(f"\n💡 [bold blue]使用示例:[/bold blue]")
        print(f"   translate -t zh     # 简体中文（默认）")
        print(f"   translate -t ja     # 日文")
        print(f"   translate -t ko     # 韩文")
        print(f"   translate -t fr     # 法文")
        raise typer.Exit(code=1)

    if output_dir is None:
        if input_dir:
            output_dir = input_dir
            logger.info(f"使用输入目录作为输出目录: {output_dir}")
        else:
            output_dir = Path.cwd()
            logger.info(f"使用当前目录作为输出目录: {output_dir}")
    else:
        logger.info(f"使用指定输出目录: {output_dir}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录已解析为: {output_dir}")

    if not output_dir.exists():
        logger.error(f"输出目录不存在: {output_dir}")
        print(f"[bold red]❌ 输出目录不存在: {output_dir}[/bold red]")
        raise typer.Exit(code=1)

    if not os.access(output_dir, os.W_OK):
        logger.error(f"输出目录不可写: {output_dir}")
        print(f"[bold red]❌ 输出目录不可写: {output_dir}[/bold red]")
        raise typer.Exit(code=1)

    if input_file:
        if input_file.suffix.lower() != '.srt' and not transcribe:
            logger.error(f"未启用转录功能，无法处理非字幕文件: {input_file.name}")
            print(f"[bold red]❌ 未启用转录功能![/bold red]")
            print(f"文件 [cyan]{input_file.name}[/cyan] 需要转录才能处理。")
            print(f"请添加 [bold magenta]--transcribe[/bold magenta] 参数以启用转录功能。")
            raise typer.Exit(code=1)

        files_to_process = [input_file]
        batch_input_dir = input_file.parent
        logger.info(f"开始处理单个文件: {input_file.name}")
        print(f"开始处理单个文件: [bold cyan]{input_file.name}[/bold cyan]")
    else:
        batch_input_dir = input_dir if input_dir else Path.cwd()
        batch_input_dir = batch_input_dir.resolve()
        files_to_process = _get_batch_files(max_count, llm_model, batch_input_dir, transcribe)

    if dry_run:
        _show_dry_run_summary(files_to_process, target_lang, output_dir, model, llm_model, batch_input_dir)
        raise typer.Exit(code=0)

    _process_files_batch(files_to_process, target_lang, output_dir, model, llm_model,
                        split_model, translation_model, preserve_intermediate)


def _validate_target_language(target_lang: str):
    """验证目标语言代码"""
    from .translation_core.config import get_target_language
    target_language_name = get_target_language(target_lang)
    print(f"🎯 [bold green]目标语言:[/bold green] [cyan]{target_language_name}[/cyan] ([dim]{target_lang}[/dim])")


def _get_file_type_info(file_ext: str) -> Tuple[str, str]:
    """获取文件类型和处理方式"""
    if file_ext == '.srt':
        return "字幕文件", "直接翻译"
    if file_ext in AUDIO_EXTENSIONS:
        return "音频文件", "转录+翻译"
    if file_ext in VIDEO_EXTENSIONS:
        return "视频文件", "转录+翻译"
    return "未知类型", "未知"


def _format_file_size(file_path: Path) -> str:
    """格式化文件大小显示"""
    try:
        file_size = file_path.stat().st_size
        if file_size < 1024:
            return f"{file_size} B"
        if file_size < 1024 * 1024:
            return f"{file_size / 1024:.1f} KB"
        return f"{file_size / (1024 * 1024):.1f} MB"
    except:
        return "未知"


def _remove_language_suffix(base_name: str) -> str:
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


def _natural_sort_key(s: str):
    """用于自然排序的key函数：将数字片段按整数比较，其他片段按不区分大小写的字符串比较"""
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.casefold() for p in parts]


def _get_batch_files(max_count: int, llm_model: Optional[str], input_dir: Path, transcribe: bool) -> list:
    """获取批量处理的文件列表"""
    if transcribe:
        # 转录模式：支持字幕和所有媒体文件
        patterns = ["*.srt"] + [f"*{ext}" for ext in MEDIA_EXTENSIONS]
    else:
        # 翻译模式：只支持字幕文件
        patterns = ["*.srt"]

    # 确保input_dir是绝对路径
    input_dir = input_dir.resolve()

    # 查找所有媒体文件（使用绝对路径）
    media_files = []
    for pattern in patterns:
        media_files.extend(glob.glob(str(input_dir / pattern)))
    
    if not media_files:
        print(f"[bold red]{input_dir} 目录中没有找到需要处理的媒体文件。[/bold red]")
        print("[dim]支持的格式：[/dim]")
        print("[dim]  • 字幕文件: .srt[/dim]")
        print("[dim]  • 音频文件: .mp3, .m4a, .wav, .flac, .aac, .ogg, .wma, .aiff, .opus[/dim]")
        print("[dim]  • 视频文件: .mp4, .avi, .mov, .mkv, .webm, .flv, .wmv, .m4v, .mpeg, .mpg, .3gp, .ts[/dim]")
        raise typer.Exit(code=1)
    
    # 提取基础文件名并去重排序
    base_names = set()
    for file_path in media_files:
        # 转换为Path对象并获取相对于input_dir的路径
        file = Path(file_path)
        relative_path = file.relative_to(input_dir)
        file_name = relative_path.name

        # 移除扩展名（使用常量构建正则表达式）
        all_exts = ['srt'] + [ext.lstrip('.') for ext in MEDIA_EXTENSIONS]
        ext_pattern = r'\.(' + '|'.join(all_exts) + r')$'
        base_name = re.sub(ext_pattern, '', file_name, flags=re.IGNORECASE)

        # 移除各种语言后缀
        base_name = _remove_language_suffix(base_name)
        base_names.add(base_name)
    
    # 自然排序基础文件名（EP2 在 EP10 之前）
    base_names = sorted(base_names, key=_natural_sort_key)
    
    # 为每个基础名称找到对应的输入文件
    files_to_process = []
    for base_name in base_names:
        # 跳过已存在.ass文件的
        ass_file = input_dir / f"{base_name}.ass"
        if ass_file.exists():
            continue

        # 确定输入文件优先级：srt > 音频 > 视频（音频转录更快）
        input_file_found = None

        for ext in ['.srt'] + AUDIO_EXTENSIONS + VIDEO_EXTENSIONS:
            candidate = input_dir / f"{base_name}{ext}"
            if candidate.exists():
                input_file_found = candidate
                break

        if input_file_found:
            files_to_process.append(input_file_found)
            print(f"📄 发现文件 [cyan]{input_file_found}[/cyan]")
        else:
            print(f"❌ 没有找到 [yellow]{base_name}[/yellow] 的输入文件")
    
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


def _show_dry_run_summary(files_to_process: list, target_lang: str, output_dir: Path,
                         model: str, llm_model: Optional[str], input_dir: Path):
    """显示预览模式的文件处理信息"""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

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

    needs_transcription = any(f.suffix.lower() != '.srt' for f in files_to_process)
    if needs_transcription:
        info_table.add_row("🎙️  转录模型", model)

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
            file_type, process_type = _get_file_type_info(file_ext)

            # 获取文件大小
            size_str = _format_file_size(file_path)

            file_table.add_row(str(idx), file_name, file_type, size_str, process_type)

        console.print(file_table)

        # 统计信息
        total_size = sum(f.stat().st_size for f in files_to_process if f.exists())
        srt_count = sum(1 for f in files_to_process if f.suffix.lower() == '.srt')
        media_count = len(files_to_process) - srt_count

        summary = f"""
[bold]📊 处理统计:[/bold]
• 总文件数: {len(files_to_process)} 个
• 字幕文件: {srt_count} 个 (直接翻译)
• 音视频文件: {media_count} 个 (转录+翻译)
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


def _process_files_batch(files_to_process: list, target_lang: str, output_dir: Path,
                        model: str, llm_model: Optional[str],
                        split_model: Optional[str],
                        translation_model: Optional[str], preserve_intermediate: bool):
    """批量处理文件"""
    from .transcription_core.model_cache import model_context
    
    count = 0
    generated_ass_files = []
    
    # 全局预检查转录模型（只对需要转录的文件执行）
    model_precheck_passed = None
    needs_transcription = any(f.suffix.lower() != '.srt' for f in files_to_process)
    
    if needs_transcription:
        from .processor import precheck_model_availability
        print("[bold blue]>>> 预检查转录环境...[/bold blue]")
        model_precheck_passed = precheck_model_availability(model, show_progress=True)
        
        if not model_precheck_passed:
            print("[bold red]❌ 转录模型不可用，无法处理需要转录的文件[/bold red]")
            # 过滤掉需要转录的文件，只处理 .srt 文件
            srt_files = [f for f in files_to_process if f.suffix.lower() == '.srt']
            if srt_files:
                print(f"[bold yellow]将只处理 {len(srt_files)} 个 SRT 文件[/bold yellow]")
                files_to_process = srt_files
            else:
                print("[bold red]没有可处理的 SRT 文件，退出批量处理[/bold red]")
                return
    
    # 在批量处理开始时初始化翻译服务并显示配置（只显示一次）
    from .service import SubtitleTranslatorService
    try:
        translator_service = SubtitleTranslatorService()
        translator_service._init_translation_env(
            llm_model=llm_model,
            split_model=split_model,
            translation_model=translation_model,
            show_config=True
        )
        print()  # 添加空行分隔
    except Exception as init_error:
        print(f"[bold red]创建翻译服务失败:[/bold red] {init_error}")
        raise
    
    # 根据文件数量决定使用批量模式还是单文件模式
    is_batch_mode = len(files_to_process) > 1
    
    with model_context(batch_mode=is_batch_mode):
        for i, current_input_file in enumerate(files_to_process):
            print()
            logger.info(f"🎯 处理文件 ({i+1}/{len(files_to_process)}): {current_input_file.name}")
            if is_batch_mode:
                print(f"🎯 [bold cyan]开始处理第 {i+1}/{len(files_to_process)} 个文件...[/bold cyan]")
            else:
                print(f"🎯 [bold cyan]开始处理文件...[/bold cyan]")
            
            try:
                # 根据实际情况传递批量模式标志
                from .processor import process_single_file
                process_single_file(
                    current_input_file, target_lang, output_dir, model,
                    llm_model, model_precheck_passed,
                    batch_mode=is_batch_mode, translator_service=translator_service,
                    preserve_intermediate=preserve_intermediate
                )
                count += 1
                
                # 检查是否生成了ASS文件
                ass_file = output_dir / f"{current_input_file.stem}.ass"
                if ass_file.exists():
                    generated_ass_files.append(ass_file)
                    logger.info(f"📺 双语ASS文件已生成: {ass_file.name}")
                    print(f"📺 [cyan]双语ASS文件已生成[/cyan]")
                
                logger.info(f"✅ {current_input_file.stem} 处理完成！")
                print(f"[bold green]✅ 处理完成！[/bold green]")
            
            except Exception as e:
                from .exceptions import SmartSplitError, TranslationError
                if isinstance(e, (SmartSplitError, TranslationError)):
                    # 这些异常已经在processor.py中显示过了，这里不重复显示
                    # 但需要记录到日志中用于统计
                    logger.info(f"❌ {current_input_file.stem} 处理失败: {e}")
                else:
                    logger.error(f"❌ {current_input_file.stem} 处理失败: {e}")
                    print(f"[bold red]❌ {current_input_file.stem} 处理失败！{e}[/bold red]")
            
            print()  # 添加空行分隔
    
    # 处理完成，显示模型优化信息
    if needs_transcription and count > 0:
        if is_batch_mode:
            print("🎯 [dim]批量处理完成，模型已自动释放，内存已优化[/dim]")
        else:
            print("🎯 [dim]处理完成，模型已自动释放，内存已优化[/dim]")
    
    # 显示处理结果
    _show_results(count, generated_ass_files, output_dir, is_batch_mode)


def _show_results(count: int, generated_ass_files: list, output_dir: Path, is_batch_mode: bool):
    """显示处理结果"""
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


@app.command("model")
def model_cmd(
    ctx: typer.Context,
    action: str = typer.Argument(..., help="要执行的操作: list(列出已缓存模型), info(显示模型信息), download(预下载模型), clean(清理缓存)"),
    model_id: Optional[str] = typer.Argument(None, help=f"模型ID (download和info操作默认: {DEFAULT_TRANSCRIPTION_MODEL})")
):
    """模型管理命令"""
    from rich.console import Console
    from rich.table import Table
    from pathlib import Path
    import os
    import shutil
    
    console = Console()
    
    if action == "list":
        """列出已缓存的模型"""
        try:
            # 获取缓存目录
            cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or Path.home() / ".cache" / "huggingface"
            cache_dir = Path(cache_dir) / "hub"
            
            if not cache_dir.exists():
                console.print("[yellow]📂 还没有缓存任何模型[/yellow]")
                return
            
            # 查找模型缓存目录
            model_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith("models--")]
            
            if not model_dirs:
                console.print("[yellow]📂 还没有缓存任何模型[/yellow]")
                return
            
            # 创建表格显示模型信息
            table = Table(title="🤖 已缓存的模型列表")
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
            console.print(f"[red]❌ 获取模型列表失败: {str(e)}[/red]")
    
    elif action == "info":
        """显示指定模型的详细信息"""
        # 如果没有指定模型ID，使用默认模型
        if not model_id:
            model_id = DEFAULT_TRANSCRIPTION_MODEL
            console.print(f"[dim]使用默认模型: {model_id}[/dim]")

        try:
            # 尝试查找本地缓存
            from .transcription_core.utils import _find_model_in_hf_cache
            try:
                model_path = _find_model_in_hf_cache(model_id)
                console.print(f"✅ [green]模型已缓存[/green]: [bold]{model_id}[/bold]")
                console.print(f"📄 模型路径: [dim]{model_path}[/dim]")

                # 显示文件大小
                config_path = model_path / "config.json"
                weight_path = model_path / "model.safetensors"
                if config_path.exists() and weight_path.exists():
                    config_size = config_path.stat().st_size / 1024
                    weight_size = weight_path.stat().st_size / (1024 * 1024)
                    console.print(f"📊 大小: 配置 {config_size:.1f} KB, 权重 {weight_size:.1f} MB")

            except FileNotFoundError:
                console.print(f"[yellow]⚠️  模型未缓存[/yellow]: [bold]{model_id}[/bold]")
                console.print(f"💡 请使用以下命令下载模型:")
                console.print(f"   [green]hf download {model_id}[/green]")

        except Exception as e:
            console.print(f"[red]❌ 获取模型信息失败: {str(e)}[/red]")
    
    elif action == "download":
        """预下载指定模型"""
        console.print(f"[yellow]⚠️  该命令已废弃[/yellow]")
        console.print(f"请使用 HuggingFace CLI 下载模型:")
        console.print(f"  [green]hf download {model_id or DEFAULT_TRANSCRIPTION_MODEL}[/green]")
    
    elif action == "clean":
        """清理模型缓存"""
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
                console.print(f"⚠️  [yellow]即将清理 {size_mb:.1f} MB 的模型缓存[/yellow]")
                console.print(f"📍 缓存位置: [dim]{cache_dir}[/dim]")
                
                confirm = typer.confirm("确定要清理所有模型缓存吗？")
                if not confirm:
                    console.print("❌ 取消清理操作")
                    return
                
                # 清理缓存
                shutil.rmtree(cache_dir)
                console.print("✅ [green]模型缓存清理完成[/green]")
            else:
                console.print("[yellow]📂 缓存目录为空，无需清理[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ 清理缓存失败: {str(e)}[/red]")
    
    else:
        console.print(f"[red]❌ 未知操作: {action}[/red]")
        console.print("💡 支持的操作: list, info, download, clean")
        console.print("\n📖 使用示例:")
        console.print("   translate model list                                    # 列出已缓存模型")
        console.print("   translate model info                                    # 显示默认模型信息")
        console.print("   translate model info mlx-community/parakeet-tdt-0.6b-v2  # 显示指定模型信息")
        console.print("   translate model download                                      # 预下载默认模型")
        console.print("   translate model download mlx-community/parakeet-tdt-0.6b-v2  # 预下载指定模型")
        console.print("   translate model clean                                   # 清理缓存")



@app.command("version")
def version():
    """显示版本信息"""
    from rich.console import Console
    from .version_utils import display_version_info

    console = Console()
    display_version_info(console)


@app.command("init")
def init():
    """初始化配置文件"""
    from pathlib import Path
    from rich import print
    from rich.prompt import Prompt, Confirm
    import os

    # 获取配置文件路径
    config_dir = Path.home() / ".config" / "subtitle-translator"
    config_file = config_dir / ".env"

    # 检查是否已存在
    if config_file.exists():
        print(f"[yellow]⚠️  配置文件已存在:[/yellow] {config_file}")
        overwrite = Confirm.ask("是否覆盖现有配置？", default=False)
        if not overwrite:
            print("[blue]ℹ️  初始化已取消[/blue]")
            return

    print("[bold green]🚀 Subtitle Translator 配置初始化[/bold green]\n")

    # 交互式输入
    print("[bold]1. API 配置[/bold]")
    api_base = Prompt.ask(
        "API Base URL",
        default="https://api.openai.com/v1"
    )
    api_key = Prompt.ask("API Key", password=True)

    print("\n[bold]2. 模型配置[/bold]")
    split_model = Prompt.ask(
        "断句模型 (用于智能分句)",
        default="gpt-4o-mini"
    )
    translation_model = Prompt.ask(
        "翻译模型 (用于字幕翻译)",
        default="gpt-4o"
    )

    # 创建配置内容
    config_content = f"""# Subtitle Translator 配置文件
# 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# API 配置
OPENAI_BASE_URL={api_base}
OPENAI_API_KEY={api_key}

# 模型配置
SPLIT_MODEL={split_model}
TRANSLATION_MODEL={translation_model}
LLM_MODEL={split_model}

# 可选配置
# TARGET_LANGUAGE=zh  # 默认目标语言
"""

    # 创建目录并写入文件
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(config_content, encoding='utf-8')

    # 设置文件权限（仅所有者可读写）
    os.chmod(config_file, 0o600)

    print(f"\n[bold green]✅ 配置文件已创建:[/bold green] {config_file}")
    print(f"\n[bold blue]💡 下一步:[/bold blue]")
    print(f"   运行 [green]translate -i your-file.srt[/green] 开始翻译")
    print(f"   或运行 [green]translate --help[/green] 查看所有选项")


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
