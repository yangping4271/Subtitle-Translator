"""
文件处理模块 - 处理单个文件的核心逻辑
"""

from pathlib import Path
from typing import List, Optional

from rich import print

from .service import SubtitleTranslatorService
from .logger import setup_logger
from .output_files import SubtitleOutputFiles
from .console_views import show_results

# 初始化logger
logger = setup_logger(__name__)


def process_batch(
    files_to_process: List[Path],
    target_lang: str,
    output_dir: Path,
    llm_model: Optional[str],
    preserve_intermediate: bool,
) -> None:
    """批量处理文件"""
    count = 0
    generated_ass_files = []
    translator_service = None

    is_batch_mode = len(files_to_process) > 1
    try:
        try:
            translator_service = SubtitleTranslatorService()
            translator_service.init_translation_env(
                llm_model=llm_model,
                show_config=True,
            )
            print()
        except Exception as init_error:
            print(f"[bold red]创建翻译服务失败:[/bold red] {init_error}")
            raise

        for i, current_input_file in enumerate(files_to_process):
            print()
            logger.info(
                f"🎯 处理文件 ({i + 1}/{len(files_to_process)}): {current_input_file.name}"
            )
            if is_batch_mode:
                print(
                    f"🎯 [bold cyan]开始翻译第 {i + 1}/{len(files_to_process)} 个文件: "
                    f"[white]{current_input_file.name}[/white][/bold cyan]"
                )
            else:
                print(
                    "[bold cyan]🎯 开始翻译: "
                    f"[white]{current_input_file.name}[/white][/bold cyan]"
                )

            try:
                output_files = process_single_file(
                    current_input_file,
                    target_lang,
                    output_dir,
                    llm_model,
                    translator_service=translator_service,
                    preserve_intermediate=preserve_intermediate,
                )
                count += 1

                ass_file = output_files.bilingual_ass
                if ass_file.exists():
                    generated_ass_files.append(ass_file)
                    logger.info(f"📺 双语ASS文件已生成: {ass_file.name}")
                    print("[cyan]📺 双语ASS文件已生成[/cyan]")

                logger.info(f"✅ {current_input_file.stem} 处理完成！")
                print("[bold green]✅ 处理完成！[/bold green]")

            except Exception as e:
                from .exceptions import (
                    SmartSplitError,
                    TranslationError,
                    SubtitleProcessError,
                )

                if isinstance(
                    e, (SmartSplitError, TranslationError, SubtitleProcessError)
                ):
                    logger.info(f"❌ {current_input_file.stem} 处理失败: {e}")
                else:
                    logger.error(f"❌ {current_input_file.stem} 处理失败: {e}")
                    print(
                        f"[bold red]❌ {current_input_file.stem} 处理失败！{e}[/bold red]"
                    )

            print()
    finally:
        if translator_service is not None:
            translator_service.close()

    show_results(count, generated_ass_files, output_dir, is_batch_mode)


def _handle_translation_error(e: Exception, logger) -> None:
    """统一处理翻译相关异常"""
    from .exceptions import (
        SmartSplitError,
        TranslationError,
        EmptySubtitleError,
        SubtitleProcessError,
    )

    error_types = {
        SmartSplitError: "智能断句失败",
        TranslationError: "翻译失败",
        EmptySubtitleError: "空文件",
        SubtitleProcessError: "字幕文件错误",
    }

    for error_type, error_name in error_types.items():
        if isinstance(e, error_type):
            logger.error(f"❌ {error_name}: {e.message}")
            if e.suggestion:
                logger.error(f"{e.suggestion}")
            print(f"[bold red]❌ {error_name}:[/bold red] {e.message}")
            if e.suggestion:
                print(f"[bold yellow]{e.suggestion}[/bold yellow]")
            raise

    # 其他异常
    logger.error(f"❌ 处理失败: {e}")
    logger.debug("详细错误信息:", exc_info=True)
    print(f"[bold red]❌ 处理失败:[/bold red] {e}")
    raise


def process_single_file(
    input_file: Path,
    target_lang: str,
    output_dir: Path,
    llm_model: Optional[str],
    translator_service: Optional[SubtitleTranslatorService] = None,
    preserve_intermediate: bool = False,
) -> SubtitleOutputFiles:
    """处理单个文件的核心逻辑"""

    # 只接受 SRT 文件
    if input_file.suffix.lower() != ".srt":
        logger.error(f"只支持 SRT 字幕文件，当前文件: {input_file.name}")
        print("[bold red]❌ 只支持 SRT 字幕文件![/bold red]")
        print(f"文件 [cyan]{input_file.name}[/cyan] 不是 SRT 格式。")
        raise RuntimeError(f"只支持 SRT 字幕文件，当前文件: {input_file.name}")

    # 使用传入的翻译服务或创建新的服务
    service_was_passed = translator_service is not None
    try:
        if translator_service is None:
            translator_service = SubtitleTranslatorService()
            translator_service.init_translation_env(llm_model, show_config=True)
        output_files = translator_service.translate_srt(
            input_srt_path=input_file,
            target_lang=target_lang,
            output_dir=output_dir,
            llm_model=llm_model,
            skip_env_init=True,
            preserve_intermediate=preserve_intermediate,
        )
        logger.info(f"ASS 文件生成成功: {output_files.bilingual_ass}")
        if output_files.intermediates_preserved:
            logger.info(f"目标语言翻译文件保存至: {output_files.target_srt}")
            logger.info(f"英文翻译文件保存至: {output_files.source_srt}")
            print("💾 [bold green]已保留中间字幕文件[/bold green]")
        else:
            print("🧹 已清理 2 个中间字幕文件")
        return output_files

    except Exception as e:
        _handle_translation_error(e, logger)
    finally:
        if not service_was_passed and translator_service is not None:
            translator_service.close()
