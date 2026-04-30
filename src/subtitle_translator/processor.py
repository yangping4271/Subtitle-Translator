"""
文件处理模块 - 处理单个文件的核心逻辑
"""
import os
from pathlib import Path
from typing import List, Optional

from rich import print

from .service import SubtitleTranslatorService
from .logger import setup_logger
from .console_views import show_results

# 初始化logger
logger = setup_logger(__name__)


def process_batch(
    files_to_process: List[Path],
    target_lang: str,
    output_dir: Path,
    llm_model: Optional[str],
    split_model: Optional[str],
    translation_model: Optional[str],
    preserve_intermediate: bool,
) -> None:
    """批量处理文件"""
    count = 0
    generated_ass_files = []
    translator_service = None

    try:
        translator_service = SubtitleTranslatorService()
        translator_service.init_translation_env(
            llm_model=llm_model,
            split_model=split_model,
            translation_model=translation_model,
            show_config=True,
        )
        print()
    except Exception as init_error:
        print(f"[bold red]创建翻译服务失败:[/bold red] {init_error}")
        raise

    is_batch_mode = len(files_to_process) > 1

    for i, current_input_file in enumerate(files_to_process):
        print()
        logger.info(f"🎯 处理文件 ({i+1}/{len(files_to_process)}): {current_input_file.name}")
        if is_batch_mode:
            print(f"🎯 [bold cyan]开始处理第 {i+1}/{len(files_to_process)} 个文件...[/bold cyan]")
        else:
            print("[bold cyan]🎯 开始处理文件...[/bold cyan]")

        try:
            process_single_file(
                current_input_file, target_lang, output_dir,
                llm_model,
                translator_service=translator_service,
                preserve_intermediate=preserve_intermediate,
            )
            count += 1

            ass_file = output_dir / f"{current_input_file.stem}.ass"
            if ass_file.exists():
                generated_ass_files.append(ass_file)
                logger.info(f"📺 双语ASS文件已生成: {ass_file.name}")
                print("[cyan]📺 双语ASS文件已生成[/cyan]")

            logger.info(f"✅ {current_input_file.stem} 处理完成！")
            print("[bold green]✅ 处理完成！[/bold green]")

        except Exception as e:
            from .exceptions import SmartSplitError, TranslationError, SubtitleProcessError
            if isinstance(e, (SmartSplitError, TranslationError, SubtitleProcessError)):
                logger.info(f"❌ {current_input_file.stem} 处理失败: {e}")
            else:
                logger.error(f"❌ {current_input_file.stem} 处理失败: {e}")
                print(f"[bold red]❌ {current_input_file.stem} 处理失败！{e}[/bold red]")

        print()

    show_results(count, generated_ass_files, output_dir, is_batch_mode)


def _handle_translation_error(e: Exception, logger) -> None:
    """统一处理翻译相关异常"""
    from .exceptions import SmartSplitError, TranslationError, EmptySubtitleError, SubtitleProcessError

    error_types = {
        SmartSplitError: "智能断句失败",
        TranslationError: "翻译失败",
        EmptySubtitleError: "空文件",
        SubtitleProcessError: "字幕文件错误",
    }

    for error_type, error_name in error_types.items():
        if isinstance(e, error_type):
            logger.error(f"❌ {error_name}: {e.message}")
            if hasattr(e, 'suggestion') and e.suggestion:
                logger.error(f"{e.suggestion}")
            print(f"[bold red]❌ {error_name}:[/bold red] {e.message}")
            if hasattr(e, 'suggestion') and e.suggestion:
                print(f"[bold yellow]{e.suggestion}[/bold yellow]")
            raise

    # 其他异常
    logger.error(f"❌ 处理失败: {e}")
    logger.debug("详细错误信息:", exc_info=True)
    print(f"[bold red]❌ 处理失败:[/bold red] {e}")
    raise


def process_single_file(
    input_file: Path, target_lang: str, output_dir: Path,
    llm_model: Optional[str],
    translator_service = None,
    preserve_intermediate: bool = False
):
    """处理单个文件的核心逻辑"""

    # 只接受 SRT 文件
    if input_file.suffix.lower() != '.srt':
        logger.error(f"只支持 SRT 字幕文件，当前文件: {input_file.name}")
        print("[bold red]❌ 只支持 SRT 字幕文件![/bold red]")
        print(f"文件 [cyan]{input_file.name}[/cyan] 不是 SRT 格式。")
        raise RuntimeError(f"只支持 SRT 字幕文件，当前文件: {input_file.name}")

    print("[bold yellow]>>> 检测到SRT文件，开始翻译...[/bold yellow]")
    temp_srt_path = input_file

    final_target_lang_path = None
    final_english_path = None

    # --- 翻译阶段 ---
    logger.info(">>> 开始翻译...")
    print("[bold green]>>> 开始翻译...[/bold green]")
    
    # 使用传入的翻译服务或创建新的服务
    service_was_passed = translator_service is not None
    if translator_service is None:
        # 单文件模式，需要创建并初始化翻译服务
        try:
            translator_service = SubtitleTranslatorService()
            translator_service.init_translation_env(llm_model, show_config=True)
        except Exception as init_error:
            print(f"[bold red]创建翻译服务失败:[/bold red] {init_error}")
            raise
    # 批量模式下，翻译服务已经初始化完成，直接使用
    try:
        final_target_lang_path = translator_service.translate_srt(
            input_srt_path=temp_srt_path,
            target_lang=target_lang,
            output_dir=output_dir,
            llm_model=llm_model,
            skip_env_init=service_was_passed  # 如果服务是传入的（批量模式），跳过环境初始化
        )
        # 确保这里正确赋值
        final_english_path = output_dir / f"{temp_srt_path.stem}.en.srt"

        logger.info(f"翻译完成，目标语言翻译文件保存至: {final_target_lang_path}")
        logger.info(f"英文翻译文件保存至: {final_english_path}")

        # --- 转换为 ASS ---
        print(">>> [bold green]生成双语ASS文件...[/bold green]")
        logger.info(">>> 正在转换为 ASS 格式...")

        # 提取 srt2ass.py 的核心逻辑
        from .translation_core.utils.ass_converter import convert_srt_to_ass

        final_ass_path = convert_srt_to_ass(final_target_lang_path, final_english_path, output_dir)
        logger.info(f"ASS 文件生成成功: {final_ass_path}")

    except Exception as e:
        _handle_translation_error(e, logger)
    finally:
        # 清理中间翻译文件
        if preserve_intermediate:
            logger.info(">>> 保留中间翻译文件...")
            preserved_files = []
            if final_target_lang_path and final_target_lang_path.exists():
                preserved_files.append(f"{target_lang} SRT")
                logger.info(f"保留中间文件: {final_target_lang_path}")
            if final_english_path and final_english_path.exists():
                preserved_files.append("英文 SRT")
                logger.info(f"保留中间文件: {final_english_path}")

            if preserved_files:
                print(f"💾 [bold green]已保留中间文件:[/bold green] {', '.join(preserved_files)}")
        else:
            logger.info(">>> 正在清理中间翻译文件...")
            cleaned_files = 0
            if final_target_lang_path and final_target_lang_path.exists():
                os.remove(final_target_lang_path)
                logger.info(f"已删除中间文件: {final_target_lang_path}")
                cleaned_files += 1
            if final_english_path and final_english_path.exists():
                os.remove(final_english_path)
                logger.info(f"已删除中间文件: {final_english_path}")
                cleaned_files += 1

            if cleaned_files > 0:
                print(f"🧹 已清理 {cleaned_files} 个中间文件") 
