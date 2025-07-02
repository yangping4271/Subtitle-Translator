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
from .logger import setup_logger

# 初始化logger
logger = setup_logger(__name__)


app = typer.Typer(
    help="一个集成了语音转录、字幕翻译和格式转换的命令行工具",
    epilog="💡 首次使用请运行: subtitle-translate init 来配置API密钥"
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_file: Optional[Path] = typer.Option(None, "--input-file", "-i", help="要处理的单个文件路径，如不指定则批量处理当前目录。", exists=True, file_okay=True, dir_okay=False, readable=True),
    max_count: int = typer.Option(-1, "--count", "-n", help="最大处理文件数量，-1表示处理所有文件。"),
    target_lang: str = typer.Option("zh", "--target_lang", "-t", help="目标翻译语言。支持的语言：zh(简体中文), zh-tw(繁体中文), ja(日文), ko(韩文), en(英文), fr(法文), de(德文), es(西班牙文), pt(葡萄牙文), ru(俄文), it(意大利文), ar(阿拉伯文), th(泰文), vi(越南文)等。"),
    output_dir: Optional[Path] = typer.Option(None, "--output_dir", "-o", help="输出文件的目录，默认为当前目录。"),
    model: str = typer.Option("mlx-community/parakeet-tdt-0.6b-v2", "--model", help="用于转录的 Parakeet MLX 模型。"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", "-m", help="用于翻译的LLM模型，默认使用配置文件中的设置。"),
    reflect: bool = typer.Option(False, "--reflect", "-r", help="启用反思翻译模式，提高翻译质量但会增加处理时间。"),
    debug: bool = typer.Option(False, "--debug", "-d", help="启用调试日志级别，显示更详细的处理信息。"),
):
    """字幕翻译工具主命令"""
    setup_environment()
    
    # 如果调用了子命令，就不执行主逻辑
    if ctx.invoked_subcommand is not None:
        return
    
    # 早期验证目标语言代码，提供友好错误信息
    try:
        _validate_target_language(target_lang)
    except ValueError as e:
        logger.error(f"❌ 命令行参数错误 - 目标语言: {str(e)}")
        print(f"[bold red]❌ 目标语言参数错误![/bold red]")
        print(str(e))
        print(f"\n💡 [bold blue]使用示例:[/bold blue]")
        print(f"   subtitle-translate -t ja  # 翻译成日文")
        print(f"   subtitle-translate -t ko  # 翻译成韩文")
        print(f"   subtitle-translate -t fr  # 翻译成法文")
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
    _process_files_batch(files_to_process, target_lang, output_dir, model, llm_model, reflect, debug)


def _validate_target_language(target_lang: str):
    """验证目标语言代码"""
    from .translation_core.config import get_target_language
    target_language_name = get_target_language(target_lang)
    print(f"🎯 [bold green]目标语言:[/bold green] [cyan]{target_language_name}[/cyan] ([dim]{target_lang}[/dim])")


def _get_batch_files(max_count: int, llm_model: Optional[str]) -> list:
    """获取批量处理的文件列表"""
    MEDIA_EXTENSIONS = ["*.srt", "*.mp3", "*.mp4"]

    # 查找所有媒体文件
    media_files = []
    for pattern in MEDIA_EXTENSIONS:
        media_files.extend(glob.glob(pattern))
    
    if not media_files:
        print("[bold red]当前目录没有找到需要处理的文件 (*.srt, *.mp3, *.mp4)。[/bold red]")
        raise typer.Exit(code=1)
    
    # 提取基础文件名并去重排序
    base_names = set()
    for file in media_files:
        # 移除扩展名
        base_name = re.sub(r'\.(srt|mp3|mp4)$', '', file)
        # 移除各种语言后缀
        language_suffixes = [r'\.zh$', r'\.zh-cn$', r'\.zh-tw$', r'\.ja$', r'\.en$', r'\.ko$', r'\.fr$', r'\.de$', r'\.es$', r'\.pt$', r'\.ru$', r'\.it$', r'\.ar$', r'\.th$', r'\.vi$']
        for suffix_pattern in language_suffixes:
            base_name = re.sub(suffix_pattern, '', base_name)
        base_names.add(base_name)
    
    base_names = sorted(base_names)
    
    # 为每个基础名称找到对应的输入文件
    files_to_process = []
    for base_name in base_names:
        # 跳过已存在.ass文件的
        ass_file = Path(f"{base_name}.ass")
        if ass_file.exists():
            print(f"INFO: {base_name}.ass 已存在，跳过处理。")
            continue
        
        # 确定输入文件优先级：srt > mp3 > mp4
        input_file_found = None
        for ext in ['.srt', '.mp3', '.mp4']:
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
                        model: str, llm_model: Optional[str], reflect: bool, debug: bool):
    """批量处理文件"""
    count = 0
    generated_ass_files = []
    
    for i, current_input_file in enumerate(files_to_process):
        print()
        logger.info(f"🎯 处理文件 ({i+1}/{len(files_to_process)}): {current_input_file.name}")
        print(f"🎯 处理文件 ({i+1}/{len(files_to_process)}): [bold cyan]{current_input_file.name}[/bold cyan]")
        
        try:
            process_single_file(
                current_input_file, target_lang, output_dir, model, 
                llm_model, reflect, debug
            )
            count += 1
            logger.info(f"✅ {current_input_file.stem} 处理完成！")
            print(f"[bold green]✅ {current_input_file.stem} 处理完成！[/bold green]")
            
            # 检查是否生成了ASS文件
            ass_file = output_dir / f"{current_input_file.stem}.ass"
            if ass_file.exists():
                generated_ass_files.append(ass_file)
                logger.info(f"📺 双语ASS文件已生成: {ass_file.name}")
                print(f"📺 双语ASS文件已生成: [cyan]{ass_file.name}[/cyan]")
        
        except Exception as e:
            from .translation_core.spliter import SmartSplitError, TranslationError, SummaryError
            if isinstance(e, (SmartSplitError, TranslationError, SummaryError)):
                # 这些异常已经在processor.py中显示过了，这里不重复显示
                pass
            else:
                print(f"[bold red]❌ {current_input_file.stem} 处理失败！{e}[/bold red]")
        
        print()  # 添加空行分隔
    
    # 显示处理结果
    _show_batch_results(count, generated_ass_files, output_dir)


def _show_batch_results(count: int, generated_ass_files: list, output_dir: Path):
    """显示批量处理结果"""
    print()
    logger.info("🎉 批量处理完成！")
    logger.info(f"总计处理文件数: {count}")
    print(f"🎉 [bold green]批量处理完成！[/bold green] (处理 [cyan]{count}[/cyan] 个文件)")
    
    # 只显示本次生成的ASS文件统计
    if count > 0:
        if generated_ass_files:
            logger.info("本次生成的ASS文件：")
            for f in generated_ass_files:
                logger.info(f"  {f.name}")
            print(f"📺 [bold green]已生成 {len(generated_ass_files)} 个双语ASS文件[/bold green]")
        
        # 过滤掉语言特定的SRT文件
        language_patterns = ['.zh.', '.zh-cn.', '.zh-tw.', '.ja.', '.en.', '.ko.', '.fr.', '.de.', '.es.', '.pt.', '.ru.', '.it.', '.ar.', '.th.', '.vi.']
        srt_files = [f for f in output_dir.glob("*.srt") if not any(pattern in f.name for pattern in language_patterns)]
        if srt_files:
            logger.info("原始字幕文件：")
            for f in srt_files:
                logger.info(f"  {f.name}")
    
    logger.info("处理完毕！")


@app.command("init")
def init():
    """初始化全局配置 - 检查当前目录.env文件或交互式输入配置"""
    init_config()


if __name__ == "__main__":
    app() 