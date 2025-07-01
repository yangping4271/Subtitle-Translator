"""
文件处理模块 - 处理单个文件的核心逻辑
"""
import os
from pathlib import Path
from typing import Optional

from rich import print

from .service import SubtitleTranslatorService
from .transcription_core import from_pretrained
from .transcription_core.cli import to_srt
from .logger import setup_logger

# 初始化logger
logger = setup_logger(__name__)


def process_single_file(
    input_file: Path, target_lang: str, output_dir: Path, 
    model: str, llm_model: Optional[str], reflect: bool, debug: bool
):
    """处理单个文件的核心逻辑"""

    # 检测输入文件类型
    if input_file.suffix.lower() == '.srt':
        print("[bold yellow]>>> 检测到SRT文件，跳过转录步骤...[/bold yellow]")
        temp_srt_path = input_file
    else:
        # --- 转录阶段 ---
        logger.info(">>> 开始转录...")
        print("[bold green]>>> 开始转录...[/bold green]")
        temp_srt_path = output_dir / f"{input_file.stem}.srt"
        try:
            # 模拟 parakeet-mlx 的转录过程
            # 实际这里需要调用 parakeet-mlx 的核心转录函数
            # 由于 parakeet-mlx 的 cli.py 中的 main 函数直接处理文件并保存，
            # 我们需要将其核心逻辑提取出来，或者直接调用其内部的 transcribe 方法。
            # 这里暂时用一个占位符，后续需要将 parakeet-mlx 的转录逻辑封装成一个可调用的函数。
            
            # 假设 from_pretrained 返回一个模型实例，并且该实例有 transcribe 方法
            # 并且 transcribe 方法返回 AlignedResult
            loaded_model = from_pretrained(model)
            
            # 对于大文件，使用分块处理避免内存溢出
            # 使用与原始parakeet-mlx相同的默认值：120秒分块，15秒重叠
            result = loaded_model.transcribe(input_file, chunk_duration=120.0, overlap_duration=15.0)
            
            # 将转录结果保存为 SRT，使用 timestamps=True 获得更精细的时间戳
            srt_content = to_srt(result, timestamps=True)
            with open(temp_srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            
            # 统计字幕数量
            subtitle_count = len(srt_content.strip().split('\n\n'))
            logger.info(f"转录完成，SRT文件保存至: {temp_srt_path}")
            print(f"✅ [bold green]转录完成[/bold green] (共 [cyan]{subtitle_count}[/cyan] 条字幕)")

        except Exception as e:
            print(f"[bold red]转录失败:[/bold red] {e}")
            raise RuntimeError(f"转录失败: {e}")

    final_target_lang_path = None
    final_english_path = None

    # --- 翻译阶段 ---
    logger.info(">>> 开始翻译...")
    print("[bold green]>>> 开始翻译...[/bold green]")
    try:
        translator_service = SubtitleTranslatorService()
    except Exception as init_error:
        print(f"[bold red]创建翻译服务失败:[/bold red] {init_error}")
        raise
    try:
        final_target_lang_path = translator_service.translate_srt(
            input_srt_path=temp_srt_path,
            target_lang=target_lang,
            output_dir=output_dir,
            llm_model=llm_model,
            reflect=reflect
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
        print(f"[bold red]翻译或 ASS 转换失败:[/bold red] {e}")
        raise RuntimeError(f"翻译或 ASS 转换失败: {e}")
    finally:
        # --- 清理中间翻译文件，保留原始转录文件 ---
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
        
        # 处理原始SRT文件
        if temp_srt_path and temp_srt_path.exists():
            if input_file.suffix.lower() != '.srt':
                logger.info(f"保留原始转录文件: {temp_srt_path}")
                print(f"💾 [bold green]保留转录文件:[/bold green] [cyan]{temp_srt_path.name}[/cyan]") 