"""
主命令行接口模块 - 简洁清晰的CLI入口
"""
import glob
import re
import typer
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

from rich import print

from .env_setup import setup_environment
from .processor import process_single_file
from .config_manager import init_config
from .logger import setup_logger, get_log_file_path, get_log_mode_info
from .transcription_core.utils import _find_cached_model, _check_network_connectivity, from_pretrained
from .transcription_core import utils as transcription_utils

# 默认转录模型
DEFAULT_TRANSCRIPTION_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"

# 初始化logger
logger = setup_logger(__name__)


app = typer.Typer(
    help="一个集成了语音转录、字幕翻译和格式转换的命令行工具",
    epilog="💡 首次使用请运行: translate init 来配置API密钥"
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_file: Optional[Path] = typer.Option(None, "--input-file", "-i", help="要处理的单个文件路径，如不指定则批量处理当前目录。", exists=True, file_okay=True, dir_okay=False, readable=True),
    max_count: int = typer.Option(-1, "--count", "-n", help="最大处理文件数量，-1表示处理所有文件。"),
    target_lang: str = typer.Option("zh", "--target_lang", "-t", help="目标翻译语言。支持：zh/zh-cn(简中), zh-tw(繁中), ja(日), ko(韩), fr(法), de(德), es(西), pt(葡), it(意), ru(俄), ar(阿), th(泰), vi(越)等。"),
    output_dir: Optional[Path] = typer.Option(None, "--output_dir", "-o", help="输出文件的目录，默认为当前目录。"),
    model: str = typer.Option(DEFAULT_TRANSCRIPTION_MODEL, "--model", help="用于转录的 Parakeet MLX 模型。"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", "-m", help="用于翻译的LLM模型，默认使用配置文件中的设置。"),
    preserve_intermediate: bool = typer.Option(False, "--preserve-intermediate", "-p", help="保留中间的英文和目标语言SRT文件，便于进一步处理或调试。"),
    version: bool = typer.Option(False, "--version", help="显示版本信息并退出。"),
):
    """字幕翻译工具主命令"""
    # 处理版本信息请求
    if version:
        from .version_utils import get_simple_version_info
        print(get_simple_version_info())
        raise typer.Exit()

    # 如果调用了子命令，就不执行主逻辑
    if ctx.invoked_subcommand is not None:
        return

    setup_environment()
    
    # 显示日志文件路径信息
    log_mode, log_location = get_log_mode_info()
    log_path = get_log_file_path()
    print(f"📝 [dim]日志模式: {log_mode} ({log_location})[/dim]")
    print(f"📝 [dim]日志文件: {log_path}[/dim]")
    print()  # 空行分隔
    
    # 早期验证目标语言代码，提供友好错误信息
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

    # 设置输出目录
    if output_dir is None:
        output_dir = Path.cwd()
    
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取要处理的文件列表
    if input_file:
        files_to_process = [input_file]
        logger.info(f"开始处理单个文件: {input_file.name}")
        print(f"开始处理单个文件: [bold cyan]{input_file.name}[/bold cyan]")
    else:
        files_to_process = _get_batch_files(max_count, llm_model)

    # 批量处理文件
    _process_files_batch(files_to_process, target_lang, output_dir, model, llm_model, preserve_intermediate)


def _validate_target_language(target_lang: str):
    """验证目标语言代码"""
    from .translation_core.config import get_target_language
    target_language_name = get_target_language(target_lang)
    print(f"🎯 [bold green]目标语言:[/bold green] [cyan]{target_language_name}[/cyan] ([dim]{target_lang}[/dim])")


def _natural_sort_key(s: str):
    """用于自然排序的key函数：将数字片段按整数比较，其他片段按不区分大小写的字符串比较"""
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.casefold() for p in parts]


def _get_batch_files(max_count: int, llm_model: Optional[str]) -> list:
    """获取批量处理的文件列表"""
    MEDIA_EXTENSIONS = [
        "*.srt",  # 字幕文件
        # 音频格式（优先级高于视频）
        "*.mp3", "*.m4a", "*.wav", "*.flac", "*.aac",
        "*.ogg", "*.wma", "*.aiff", "*.opus",
        # 视频格式
        "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm",
        "*.flv", "*.wmv", "*.m4v", "*.mpeg", "*.mpg",
        "*.3gp", "*.ts"
    ]

    # 查找所有媒体文件
    media_files = []
    for pattern in MEDIA_EXTENSIONS:
        media_files.extend(glob.glob(pattern))
    
    if not media_files:
        print("[bold red]当前目录没有找到需要处理的媒体文件。[/bold red]")
        print("[dim]支持的格式：[/dim]")
        print("[dim]  • 字幕文件: .srt[/dim]")
        print("[dim]  • 音频文件: .mp3, .m4a, .wav, .flac, .aac, .ogg, .wma, .aiff, .opus[/dim]")
        print("[dim]  • 视频文件: .mp4, .avi, .mov, .mkv, .webm, .flv, .wmv, .m4v, .mpeg, .mpg, .3gp, .ts[/dim]")
        raise typer.Exit(code=1)
    
    # 提取基础文件名并去重排序
    base_names = set()
    for file in media_files:
        # 移除扩展名
        base_name = re.sub(
            r'\.(srt|mp3|m4a|wav|flac|aac|ogg|wma|aiff|opus|'
            r'mp4|avi|mov|mkv|webm|flv|wmv|m4v|mpeg|mpg|3gp|ts)$',
            '', file, flags=re.IGNORECASE
        )
        # 移除各种语言后缀
        language_suffixes = [
            r'\.zh$', r'\.zh-cn$', r'\.zh-tw$',  # 中文
            r'\.ja$', r'\.ko$', r'\.th$', r'\.vi$',  # 亚洲语言
            r'\.fr$', r'\.de$', r'\.es$', r'\.pt$', r'\.it$', r'\.ru$',  # 欧洲语言
            r'\.ar$', r'\.en$'  # 其他
        ]
        for suffix_pattern in language_suffixes:
            base_name = re.sub(suffix_pattern, '', base_name)
        base_names.add(base_name)
    
    # 自然排序基础文件名（EP2 在 EP10 之前）
    base_names = sorted(base_names, key=_natural_sort_key)
    
    # 为每个基础名称找到对应的输入文件
    files_to_process = []
    for base_name in base_names:
        # 跳过已存在.ass文件的
        ass_file = Path(f"{base_name}.ass")
        if ass_file.exists():
            print(f"INFO: {base_name}.ass 已存在，跳过处理。")
            continue
        
        # 确定输入文件优先级：srt > 音频 > 视频（音频转录更快）
        input_file_found = None
        # 音频格式列表（按常用程度排序）
        audio_exts = ['.mp3', '.m4a', '.wav', '.flac', '.aac',
                      '.ogg', '.wma', '.aiff', '.opus']
        # 视频格式列表（按常用程度排序）
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm',
                      '.flv', '.wmv', '.m4v', '.mpeg', '.mpg',
                      '.3gp', '.ts']

        for ext in ['.srt'] + audio_exts + video_exts:
            candidate = Path(f"{base_name}{ext}")
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


def _process_files_batch(files_to_process: list, target_lang: str, output_dir: Path,
                        model: str, llm_model: Optional[str], preserve_intermediate: bool):
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
        translator_service._init_translation_env(llm_model, show_config=True)
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
                from .translation_core.spliter import SmartSplitError, TranslationError, SummaryError
                if isinstance(e, (SmartSplitError, TranslationError, SummaryError)):
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
        
        # 过滤掉语言特定的SRT文件
        language_patterns = [
            '.zh.', '.zh-cn.', '.zh-tw.',  # 中文
            '.ja.', '.ko.', '.th.', '.vi.',  # 亚洲语言
            '.fr.', '.de.', '.es.', '.pt.', '.it.', '.ru.',  # 欧洲语言
            '.ar.', '.en.'  # 其他
        ]
        srt_files = [f for f in output_dir.glob("*.srt") if not any(pattern in f.name for pattern in language_patterns)]
        if srt_files:
            logger.info("原始字幕文件：")
            for f in srt_files:
                logger.info(f"  {f.name}")
    
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
            try:
                config_path, weight_path = _find_cached_model(model_id)
                console.print(f"✅ [green]模型已缓存[/green]: [bold]{model_id}[/bold]")
                console.print(f"📄 配置文件: [dim]{config_path}[/dim]")
                console.print(f"⚖️  权重文件: [dim]{weight_path}[/dim]")
                
                # 显示文件大小
                config_size = Path(config_path).stat().st_size / 1024
                weight_size = Path(weight_path).stat().st_size / (1024 * 1024)
                console.print(f"📊 大小: 配置 {config_size:.1f} KB, 权重 {weight_size:.1f} MB")
                
            except FileNotFoundError:
                console.print(f"[yellow]⚠️  模型未缓存[/yellow]: [bold]{model_id}[/bold]")
                console.print("💡 你可以使用 'translate model download' 命令预下载模型")
                
                # 检查网络连接
                if _check_network_connectivity():
                    console.print("🌐 网络连接正常，模型将在首次使用时自动下载")
                else:
                    console.print("[red]🌐 网络连接异常，无法下载模型[/red]")
                    
        except Exception as e:
            console.print(f"[red]❌ 获取模型信息失败: {str(e)}[/red]")
    
    elif action == "download":
        """预下载指定模型"""
        # 如果没有指定模型ID，使用默认模型
        if not model_id:
            model_id = DEFAULT_TRANSCRIPTION_MODEL
            console.print(f"[dim]使用默认模型: {model_id}[/dim]")
        
        try:
            console.print(f"🚀 开始预下载模型: [bold]{model_id}[/bold]")
            
            # 检查是否已经缓存
            try:
                _find_cached_model(model_id)
                console.print(f"✅ [green]模型已存在于本地缓存[/green]")
                return
            except FileNotFoundError:
                pass
            
            # 下载模型
            model = from_pretrained(model_id, show_progress=True)
            console.print(f"\n🎉 [bold green]模型预下载完成![/bold green]")
            console.print(f"📍 模型已保存到本地缓存，后续使用时将直接加载")
            
        except Exception as e:
            console.print(f"[red]❌ 模型下载失败: {str(e)}[/red]")
    
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
    """初始化全局配置 - 检查当前目录.env文件或交互式输入配置"""
    import traceback
    print("🚀 开始初始化配置...")
    try:
        # 设置环境时允许缺少配置
        setup_environment(allow_missing_config=True)
        init_config()
        print("✅ 配置初始化完成")
    except Exception as e:
        logger.error(f"配置初始化失败: {e}")
        print(f"[bold red]❌ 配置初始化失败: {e}[/bold red]")
        print(f"[bold red]详细错误信息:[/bold red]")
        traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app() 
