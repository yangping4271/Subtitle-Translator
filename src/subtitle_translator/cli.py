"""
主命令行接口模块 - 简洁清晰的CLI入口
"""
import os
import typer
from pathlib import Path
from typing import Optional

from rich import print

from .env_setup import setup_environment
from .exceptions import ConfigurationError
from .logger import setup_logger
from .file_discovery import get_batch_files
from .console_views import show_dry_run_summary, show_results

logger = setup_logger(__name__)


app = typer.Typer(
    help="字幕翻译命令行工具"
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_file: Optional[Path] = typer.Option(None, "--input-file", "-i", help="要处理的单个文件路径，如不指定则批量处理当前目录或指定目录。", exists=True, file_okay=True, dir_okay=False, readable=True),
    input_dir: Optional[Path] = typer.Option(None, "--input-dir", help="批量处理时指定输入目录，不指定则使用当前目录。", exists=True, file_okay=False, dir_okay=True, readable=True),
    max_count: int = typer.Option(-1, "--count", "-n", help="最大处理文件数量，-1表示处理所有文件。"),
    target_lang: str = typer.Option("zh", "--target-lang", "-t", help="目标翻译语言。支持：zh/zh-cn(简中), zh-tw(繁中), ja(日), ko(韩), fr(法), de(德), es(西), pt(葡), it(意), ru(俄), ar(阿), th(泰), vi(越)等。"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="输出文件的目录，默认为当前目录。"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", "-m", help="覆盖所有模型（优先级低于独立参数）"),
    split_model: Optional[str] = typer.Option(None, "--split-model", help="断句模型"),
    translation_model: Optional[str] = typer.Option(None, "--translation-model", help="翻译模型"),
    preserve_intermediate: bool = typer.Option(False, "--preserve-intermediate", "-p", help="保留中间的英文和目标语言SRT文件，便于进一步处理或调试。"),
    dry_run: bool = typer.Option(False, "--dry-run", help="预览模式，只显示将要处理的文件信息而不实际执行翻译。"),
    version: bool = typer.Option(False, "--version", help="显示版本信息并退出。"),
):
    """字幕翻译工具主命令"""
    if version:
        from .version_utils import get_simple_version_info
        print(get_simple_version_info())
        raise typer.Exit()

    if ctx.invoked_subcommand is not None:
        return

    try:
        setup_environment()
    except ConfigurationError:
        raise typer.Exit(code=1)

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
        if input_file.suffix.lower() != '.srt':
            logger.error(f"只支持 SRT 字幕文件: {input_file.name}")
            print(f"[bold red]❌ 只支持 SRT 字幕文件![/bold red]")
            print(f"文件 [cyan]{input_file.name}[/cyan] 不是 SRT 格式。")
            raise typer.Exit(code=1)

        files_to_process = [input_file]
        batch_input_dir = input_file.parent
        logger.info(f"开始处理单个文件: {input_file.name}")
        print(f"开始处理单个文件: [bold cyan]{input_file.name}[/bold cyan]")
    else:
        batch_input_dir = input_dir if input_dir else Path.cwd()
        batch_input_dir = batch_input_dir.resolve()
        files_to_process = get_batch_files(max_count, llm_model, batch_input_dir)

    if dry_run:
        show_dry_run_summary(files_to_process, target_lang, output_dir, llm_model, batch_input_dir)
        raise typer.Exit(code=0)

    _process_files_batch(files_to_process, target_lang, output_dir, llm_model,
                        split_model, translation_model, preserve_intermediate)


def _validate_target_language(target_lang: str):
    """验证目标语言代码"""
    from .translation_core.config import get_target_language
    target_language_name = get_target_language(target_lang)
    print(f"🎯 [bold green]目标语言:[/bold green] [cyan]{target_language_name}[/cyan] ([dim]{target_lang}[/dim])")


def _process_files_batch(files_to_process: list, target_lang: str, output_dir: Path,
                        llm_model: Optional[str],
                        split_model: Optional[str],
                        translation_model: Optional[str], preserve_intermediate: bool):
    """批量处理文件"""
    count = 0
    generated_ass_files = []

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

    for i, current_input_file in enumerate(files_to_process):
        print()
        logger.info(f"🎯 处理文件 ({i+1}/{len(files_to_process)}): {current_input_file.name}")
        if is_batch_mode:
            print(f"🎯 [bold cyan]开始处理第 {i+1}/{len(files_to_process)} 个文件...[/bold cyan]")
        else:
            print(f"🎯 [bold cyan]开始处理文件...[/bold cyan]")

        try:
            # 处理单个文件
            from .processor import process_single_file
            process_single_file(
                current_input_file, target_lang, output_dir,
                llm_model,
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

    # 显示处理结果
    show_results(count, generated_ass_files, output_dir, is_batch_mode)


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
    except (SystemExit, ConfigurationError):
        raise
    except Exception as e:
        print(f"[bold red]❌ 发生错误:[/bold red] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    cli_main()
