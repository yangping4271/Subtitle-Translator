"""
文件处理模块 - 处理单个文件的核心逻辑
"""
import os
import time
from pathlib import Path
from typing import Optional

from rich import print

from .service import SubtitleTranslatorService
from .transcription_core import from_pretrained
from .transcription_core.cli import to_srt
from .transcription_core.model_cache import model_context
from .logger import setup_logger

# 初始化logger
logger = setup_logger(__name__)


def _handle_translation_error(e: Exception, logger) -> None:
    """统一处理翻译相关异常"""
    from .exceptions import SmartSplitError, TranslationError, EmptySubtitleError

    error_types = {
        SmartSplitError: "智能断句失败",
        TranslationError: "翻译失败",
        EmptySubtitleError: "空文件"
    }

    for error_type, error_name in error_types.items():
        if isinstance(e, error_type):
            logger.error(f"❌ {error_name}: {e.message}")
            if hasattr(e, 'suggestion') and e.suggestion:
                logger.error(f"{e.suggestion}")
            print(f"[bold red]❌ {error_name}:[/bold red] {e.message}")
            if hasattr(e, 'suggestion') and e.suggestion:
                print(f"[bold yellow]{e.suggestion}[/bold yellow]")

            # 空文件异常特殊处理
            if isinstance(e, EmptySubtitleError):
                raise RuntimeError(f"{e.message}")
            raise error_type(e.message, e.suggestion if hasattr(e, 'suggestion') else None)

    # 其他异常
    logger.error(f"❌ 处理失败: {e}")
    logger.debug("详细错误信息:", exc_info=True)
    print(f"[bold red]❌ 处理失败:[/bold red] {e}")
    raise RuntimeError(f"处理失败: {e}")


def precheck_model_availability(model: str, show_progress: bool = True, silent: bool = False) -> bool:
    """
    预检查模型可用性，确保在开始处理前模型已可用
    
    Args:
        model: 模型ID或路径
        show_progress: 是否显示进度信息
        silent: 是否静默模式（不显示任何输出）
        
    Returns:
        bool: 模型是否可用
    """
    try:
        if show_progress and not silent:
            print(f"\n🔍 [bold blue]检查转录模型可用性:[/bold blue] [cyan]{model}[/cyan]")
        
        # 尝试加载模型（但不实际使用，只是验证可用性）
        from .transcription_core.utils import _find_cached_model, _check_network_connectivity
        
        # 检查指定模型的可用性
        try:
            config_path, weight_path = _find_cached_model(model)
            if show_progress and not silent:
                print("✅ [green]模型已在本地缓存，可立即使用[/green]")
                print(f"💡 [dim]配置文件: {Path(config_path).name}[/dim]")
                print(f"💡 [dim]权重文件: {Path(weight_path).name} ({Path(weight_path).stat().st_size / 1024 / 1024:.1f}MB)[/dim]")
            return True
        except:
            # 本地没有指定模型的缓存，检查网络连接
            if show_progress and not silent:
                print("📥 [yellow]转录模型需要下载[/yellow]")
                print("   模型: mlx-community/parakeet-tdt-0.6b-v2")
                print("   大小: ~1.2GB")
                print("   说明: 首次使用需下载，后续将使用缓存")
                print()
                print("🔍 [dim]检查网络连接...[/dim]")

            if not _check_network_connectivity():
                if show_progress and not silent:
                    print("❌ [red]网络连接失败，无法下载模型[/red]")
                    print()
                    print("💡 [bold blue]解决方法:[/bold blue]")
                    print("   1. 检查网络连接是否正常")
                    print("   2. 确认可以访问 huggingface.co")
                    print("   3. 如有代理，请确保已正确配置")
                    print()
                return False

            if show_progress and not silent:
                print("✅ [green]网络正常，处理时将自动下载模型[/green]")
                print()
            return True
            
    except Exception as e:
        if show_progress and not silent:
            print(f"⚠️  [yellow]模型可用性检查失败: {e}[/yellow]")
            print("💡 [dim]将在处理时尝试下载模型[/dim]")
        return True  # 即使检查失败也继续，让实际处理时处理错误


def _check_model_precheck(model_precheck_passed: Optional[bool], model: str) -> None:
    """检查模型预检查结果，如果失败则抛出异常"""
    if model_precheck_passed is None:
        # 单文件处理模式，需要完整的预检查
        print("[bold blue]>>> 预检查转录环境...[/bold blue]")
        model_available = precheck_model_availability(model, show_progress=True)

        if not model_available:
            print("[bold red]❌ 转录模型不可用，无法继续处理[/bold red]")
            raise RuntimeError(f"转录模型 {model} 不可用")
    elif not model_precheck_passed:
        # 全局预检查失败，抛出异常
        print("[bold red]❌ 转录模型不可用，无法继续处理[/bold red]")
        raise RuntimeError(f"转录模型 {model} 不可用")


def process_single_file(
    input_file: Path, target_lang: str, output_dir: Path,
    model: str, llm_model: Optional[str],
    model_precheck_passed: Optional[bool] = None,
    batch_mode: bool = False, translator_service = None,
    preserve_intermediate: bool = False
):
    """处理单个文件的核心逻辑"""

    # 检测输入文件类型
    if input_file.suffix.lower() == '.srt':
        print("[bold yellow]>>> 检测到SRT文件，跳过转录步骤...[/bold yellow]")
        temp_srt_path = input_file
    else:
        # 检查模型预检查结果
        _check_model_precheck(model_precheck_passed, model)

        # --- 转录阶段 ---
        logger.info(">>> 开始转录...")
        print("[bold green]>>> 开始转录...[/bold green]")
        temp_srt_path = output_dir / f"{input_file.stem}.srt"
        try:
            # 转录阶段 - 使用优化的缓存管理
            logger.info("开始转录音频...")
            print(f"🎤 [bold cyan]正在转录音频...[/bold cyan]")

            # 记录转录开始时间
            transcribe_start_time = time.time()

            # 直接使用模型，不再嵌套 model_context（因为外部已有上下文管理）
            # 懒加载模型，只在需要时加载
            loaded_model = from_pretrained(
                model,
                use_cache=True  # 启用缓存优化
            )

            # 对于长音频，启用智能分块，并增大重叠以降低边界丢词风险
            # chunk_duration=-1 表示自动选择（见 parakeet.get_optimal_chunk_duration）
            # 默认启用 VAD 智能分块，获得更好的转录质量
            result = loaded_model.transcribe(
                input_file,
                chunk_duration=-1,
                overlap_duration=30.0,
                use_vad=True,  # 默认使用 VAD 智能分块
            )

            # 计算转录耗时
            transcribe_elapsed = time.time() - transcribe_start_time

            # 根据批量模式决定是否显示缓存释放信息
            if not batch_mode:
                # 此时模型缓存已自动释放
                pass  # 在单文件模式下会显示释放信息

            # 将转录结果保存为 SRT，使用 timestamps=True 获得更精细的时间戳
            srt_content = to_srt(result, timestamps=True)

            # 确保输出目录存在
            temp_srt_path.parent.mkdir(parents=True, exist_ok=True)

            # 验证输出路径可写
            if not temp_srt_path.parent.exists():
                raise RuntimeError(f"转录输出目录不存在: {temp_srt_path.parent}")

            logger.info(f"转录SRT将保存到: {temp_srt_path}")
            with open(temp_srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            logger.info(f"转录SRT已保存: {temp_srt_path}")

            # 统计字幕数量并显示时间统计
            sentence_count = len(result.sentences)
            logger.info(f"转录完成，SRT文件保存至: {temp_srt_path}")
            logger.info(f"⏱️  转录耗时: {transcribe_elapsed:.1f}秒")
            print(f"✅ [bold green]转录完成[/bold green] (共 [cyan]{sentence_count}[/cyan] 条字幕) - 耗时: [cyan]{transcribe_elapsed:.1f}秒[/cyan]")

        except Exception as e:
            print(f"[bold red]转录失败:[/bold red] {e}")
            raise RuntimeError(f"转录失败: {e}")

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
            translator_service._init_translation_env(llm_model, show_config=True)
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
        # --- 清理中间翻译文件，保留原始转录文件 ---
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
        
        # 处理原始SRT文件
        if temp_srt_path and temp_srt_path.exists():
            if input_file.suffix.lower() != '.srt':
                logger.info(f"保留原始转录文件: {temp_srt_path}")
                print(f"💾 [bold green]保留转录文件[/bold green]") 
