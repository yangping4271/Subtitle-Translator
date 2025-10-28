"""
字幕翻译服务模块 - 核心翻译服务类
"""
import time
import string
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich import print

from .translation_core.optimizer import SubtitleOptimizer
from .translation_core.summarizer import SubtitleSummarizer
from .translation_core.spliter import merge_segments
from .translation_core.config import SubtitleConfig
from .translation_core.data import SubtitleData
from .translation_core.utils.test_openai import test_openai
from .logger import setup_logger, log_section_start, log_section_end, log_stats
from .env_setup import OpenAIAPIError


class SubtitleTranslatorService:
    """字幕翻译服务类"""
    
    def __init__(self):
        self.config = SubtitleConfig()
        self.summarizer = SubtitleSummarizer(self.config)
        # 延迟初始化logger，在setup_environment中初始化
        self.logger = None

    def _get_logger(self):
        """获取logger实例"""
        if self.logger is None:
            from .env_setup import logger
            self.logger = logger
        return self.logger

    def _init_translation_env(self, llm_model: str, show_config: bool = True) -> None:
        """初始化翻译环境配置"""
        logger = self._get_logger()
        start_time = time.time()
        log_section_start(logger, "翻译环境初始化", "⚙️")
        
        if llm_model:
            self.config.split_model = llm_model
            self.config.summary_model = llm_model
            self.config.translation_model = llm_model

        logger.info(f"🌐 API端点: {self.config.openai_base_url}")
        
        model_config = {
            "断句模型": self.config.split_model,
            "总结模型": self.config.summary_model,
            "翻译模型": self.config.translation_model
        }
        log_stats(logger, model_config, "模型配置")
        
        # 只在需要时显示 API 配置
        if show_config:
            print(f"🌐 [bold blue]API 配置:[/bold blue]")
            print(f"   端点: [cyan]{self.config.openai_base_url}[/cyan]")
            # 对 API 密钥进行脱敏处理
            api_key = self.config.openai_api_key
            if api_key:
                if len(api_key) > 12:
                    # 对于长密钥，显示前6个和后6个字符，中间用 * 省略
                    masked_key = f"{api_key[:6]}{'*' * 8}{api_key[-6:]}"
                else:
                    # 对于短密钥，全部用 * 替代
                    masked_key = '*' * len(api_key)
                print(f"   密钥: [cyan]{masked_key}[/cyan]")
            else:
                print(f"   密钥: [red]未设置[/red]")
            
            # 显示模型配置
            print(f"🤖 [bold blue]模型配置:[/bold blue]")
            print(f"   断句: [cyan]{self.config.split_model}[/cyan]")
            print(f"   总结: [cyan]{self.config.summary_model}[/cyan]")
            print(f"   翻译: [cyan]{self.config.translation_model}[/cyan]")
        
        elapsed_time = time.time() - start_time
        log_section_end(logger, "翻译环境初始化", elapsed_time, "✅")

    def translate_srt(self, input_srt_path: Path, target_lang: str, output_dir: Path,
                      llm_model: Optional[str] = None, skip_env_init: bool = False) -> Path:
        """翻译字幕文件"""
        logger = self._get_logger()
        try:
            task_start_time = time.time()
            log_section_start(logger, "字幕翻译任务", "🎬")
            
            # 用于收集各阶段耗时的字典
            stage_times = {}
            
            # 设置目标语言（带友好错误处理）
            logger.info(f"🌍 设置目标语言: {target_lang}")
            try:
                self.config.set_target_language(target_lang)
                logger.info(f"✅ 目标语言已设置为: {self.config.target_language}")
            except ValueError as e:
                # 记录详细的错误信息到日志
                logger.error(f"❌ 语言设置失败: {str(e)}")
                # 为用户显示友好的错误信息
                print(f"[bold red]❌ 语言设置失败![/bold red]")
                print(str(e))
                raise ValueError(str(e))
            
            # 只在需要时初始化翻译环境
            if not skip_env_init:
                self._init_translation_env(llm_model)
            
            # 加载字幕文件
            from .translation_core.data import load_subtitle
            logger.info("📂 正在加载字幕文件...")
            asr_data = load_subtitle(str(input_srt_path))
            logger.info(f"📊 字幕统计: 共 {len(asr_data.segments)} 条字幕")
            logger.info(f"字幕内容预览: {asr_data.to_txt()[:100]}...")  
            
            # 检查字幕是否为空
            if len(asr_data.segments) == 0:
                logger.info("⚠️  SRT文件为空，跳过翻译处理")
                print(f"[yellow]⚠️  SRT文件为空，跳过翻译处理[/yellow]")
                # 使用专门的空文件异常，避免显示堆栈跟踪
                from .translation_core.spliter import EmptySubtitleError
                raise EmptySubtitleError("SRT文件为空，无法进行翻译")
            
            print(f"📊 [bold blue]加载完成[/bold blue]")
            
            # 并行预处理阶段：断句和总结同时进行（v0.5.x 性能优化）
            # 借鉴VideoCaptioner的解决方案：统一转换为单词级别后进行断句
            # 优势：1) 复用现有批量框架 2) 无额外API成本 3) 时间戳精确分配 4) 并行处理节省时间
            preprocessing_start_time = time.time()
            log_section_start(logger, "并行预处理阶段", "⚡")

            print(f"⚡ [bold cyan]启动并行预处理：断句 + 内容分析...[/bold cyan]")

            # 准备原始字幕内容用于总结（断句前）
            original_subtitle_content = asr_data.to_txt()

            # 启动断句任务
            def execute_splitting(asr_data_copy: SubtitleData) -> Tuple[SubtitleData, float]:
                """执行断句处理的任务函数"""
                section_start_time = time.time()
                log_section_start(logger, "字幕断句处理", "✂️")

                # 检查字幕类型并统一转换为单词级别
                if asr_data_copy.is_word_timestamp():
                    print(f"✂️ [bold yellow]检测到单词级别字幕，进行智能断句...[/bold yellow]")
                    logger.info("检测到单词级别时间戳，执行合并断句")
                else:
                    print(f"✂️ [bold yellow]检测到片段级别字幕，转换为单词级别后进行断句...[/bold yellow]")
                    logger.info("检测到片段级别时间戳，先转换为单词级别")
                    # 统一转换为单词级别字幕（核心创新功能）
                    # 使用音素级时间戳分配，支持多语言处理
                    asr_data_copy = asr_data_copy.split_to_word_segments()
                    logger.info(f"转换完成，生成 {len(asr_data_copy.segments)} 个单词级别片段")

                # 执行统一的断句处理流程
                # 现在所有字幕都是单词级别，可以使用相同的批量处理策略
                model = self.config.split_model
                logger.info(f"🤖 使用模型: {model}")
                logger.info(f"📏 句子长度限制: {self.config.max_word_count_english} 字")

                asr_data_copy = merge_segments(asr_data_copy, model=model,
                                             num_threads=self.config.thread_num,
                                             save_split=None)

                split_time = time.time() - section_start_time
                log_section_end(logger, "字幕断句处理", split_time, "✅")
                print(f"✅ [bold green]断句完成[/bold green] (优化为 [cyan]{len(asr_data_copy.segments)}[/cyan] 句)")

                return asr_data_copy, split_time

            # 启动总结任务
            def execute_summarization(subtitle_content: str, input_file: str) -> Tuple[dict, float]:
                """执行总结处理的任务函数"""
                summary_start_time = time.time()
                summarize_result = self._get_subtitle_summary(subtitle_content, input_file, is_parallel=True)
                summary_time = time.time() - summary_start_time
                return summarize_result, summary_time

            # 并行执行断句和总结任务
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 提交任务
                split_future = executor.submit(execute_splitting, asr_data)
                summary_future = executor.submit(execute_summarization, original_subtitle_content, str(input_srt_path.resolve()))

                # 等待任务完成并处理结果
                try:
                    # 获取断句结果
                    asr_data, split_time = split_future.result()
                    stage_times["✂️  智能断句"] = split_time

                    # 获取总结结果
                    summarize_result, summary_time = summary_future.result()
                    stage_times["🔍 内容分析"] = summary_time

                except Exception as e:
                    # 处理并行任务中的异常
                    logger.error(f"❌ 并行预处理任务失败: {str(e)}")

                    # 安全地检查任务异常
                    split_exception = None
                    summary_exception = None

                    try:
                        split_exception = split_future.exception()
                    except Exception:
                        pass

                    try:
                        summary_exception = summary_future.exception()
                    except Exception:
                        pass

                    if split_exception:
                        error_msg = f"断句任务失败: {split_exception}"
                        logger.error(f"❌ {error_msg}")
                        from .translation_core.spliter import SmartSplitError
                        raise SmartSplitError(error_msg) from split_exception

                    if summary_exception:
                        error_msg = f"内容分析任务失败: {summary_exception}"
                        logger.error(f"❌ {error_msg}")
                        from .translation_core.spliter import SummaryError
                        raise SummaryError(error_msg) from summary_exception

                    # 未知异常
                    raise e

            preprocessing_time = time.time() - preprocessing_start_time
            log_section_end(logger, "并行预处理阶段", preprocessing_time, "🎉")
            print(f"🎉 [bold green]并行预处理完成[/bold green] (总耗时: [cyan]{preprocessing_time:.1f}s[/cyan])")

            # 添加并行处理统计
            stage_times["⚡ 并行预处理"] = preprocessing_time
            
            # 翻译字幕
            translate_start_time = time.time()
            translate_result = self._translate_subtitles(asr_data, summarize_result)
            translate_time = time.time() - translate_start_time
            stage_times["🌍 常规翻译"] = translate_time
            
            # 保存字幕
            logger.info("💾 正在保存翻译结果...")
            base_name = input_srt_path.stem
            target_lang_output_path = output_dir / f"{base_name}.{target_lang}.srt"
            english_output_path = output_dir / f"{base_name}.en.srt"

            asr_data.save_translations_to_files(
                translate_result,
                str(english_output_path),
                str(target_lang_output_path)
            )
            
            total_elapsed = time.time() - task_start_time
            
            # 显示时间统计
            print()
            self._format_time_stats(stage_times, total_elapsed)
            
            # 任务完成统计
            final_stats = {
                "输入文件": input_srt_path.name,
                "字幕数量": len(asr_data.segments),
                "目标语言": target_lang,
                "总耗时": f"{total_elapsed:.1f}秒"
            }
            log_stats(logger, final_stats, "任务完成统计")
            log_section_end(logger, "字幕翻译任务", total_elapsed, "🎉")
            
            return target_lang_output_path
                
        except OpenAIAPIError as e:
            logger.error(f"🚨 API错误: {str(e)}")
            raise
        
        except Exception as e:
            # 检查是否是智能断句、翻译、总结或空文件异常，如果是则直接传播
            from .translation_core.spliter import SmartSplitError, TranslationError, SummaryError, EmptySubtitleError
            if isinstance(e, (SmartSplitError, TranslationError, SummaryError, EmptySubtitleError)):
                raise e
            
            logger.error(f"💥 处理过程中发生错误: {str(e)}")
            logger.exception("详细错误信息:")
            raise

    def _get_subtitle_summary(self, subtitle_content: str, input_file: str, is_parallel: bool = False) -> dict:
        """获取字幕内容摘要

        Args:
            subtitle_content: 字幕内容文本
            input_file: 输入文件路径
            is_parallel: 是否为并行调用模式
        """
        logger = self._get_logger()

        # 在并行模式下，不重复输出日志头部信息
        if not is_parallel:
            print(f"🔍 [bold cyan]内容分析中...[/bold cyan]")

        logger.info(f"🤖 使用模型: {self.config.summary_model}")
        summarize_result = self.summarizer.summarize(subtitle_content, input_file)
        logger.info(f"总结字幕内容:\n{summarize_result.get('summary')}\n")

        # 在并行模式下，不重复输出完成信息
        if not is_parallel:
            print(f"✅ [bold green]内容分析完成[/bold green]")

        return summarize_result

    def _translate_subtitles(self, asr_data: SubtitleData, summarize_result: dict) -> list:
        """翻译字幕内容"""
        logger = self._get_logger()
        section_start_time = time.time()
        log_section_start(logger, "字幕翻译", "🌍")

        print(f"🌍 [bold magenta]翻译中...[/bold magenta]")

        logger.info(f"🤖 使用模型: {self.config.translation_model}")
        logger.info(f"⚡ 线程数: {self.config.thread_num}")

        try:
            translator = SubtitleOptimizer(config=self.config)
            translate_result = translator.translate(asr_data, summarize_result)

            # 获取优化统计
            stats = self._get_optimization_stats(translator.batch_logs)

            section_elapsed = time.time() - section_start_time
            log_section_end(logger, "字幕翻译", section_elapsed, "🎉")
            print(f"✅ [bold green]翻译完成[/bold green]")
            
            # 显示优化统计
            if stats['total_changes'] > 0:
                print(f"📊 [bold blue]优化统计:[/bold blue]")
                if stats['format_changes'] > 0:
                    print(f"   格式优化: [cyan]{stats['format_changes']}[/cyan] 项")
                if stats['content_changes'] > 0:
                    print(f"   内容修改: [cyan]{stats['content_changes']}[/cyan] 项")
                if stats['wrong_changes'] > 0:
                    print(f"   [yellow]可疑替换: {stats['wrong_changes']} 项[/yellow]")
                print(f"   总计: [cyan]{stats['total_changes']}[/cyan] 项优化")
            else:
                print("📊 [dim]无需优化调整[/dim]")
            
            return translate_result
        except Exception as e:
            # 不在这里记录错误信息，避免重复显示  
            # 错误信息已经在processor.py中处理过了
            raise

    def _get_optimization_stats(self, batch_logs: list) -> dict:
        """从batch_logs中获取优化统计信息"""
        
        def is_format_change_only(original, optimized):
            """判断是否只有格式变化（大小写和标点符号）"""
            # 忽略大小写和标点符号后比较
            original_normalized = original.lower().translate(str.maketrans('', '', string.punctuation))
            optimized_normalized = optimized.lower().translate(str.maketrans('', '', string.punctuation))
            return original_normalized == optimized_normalized

        def is_wrong_replacement(original, optimized):
            """检测是否存在错误的替换（替换了不相关的词）"""
            import re
            # 提取所有单词
            original_words = set(re.findall(r'\b\w+\b', original.lower()))
            optimized_words = set(re.findall(r'\b\w+\b', optimized.lower()))
            # 找出被替换的词
            removed_words = original_words - optimized_words
            added_words = optimized_words - original_words
            # 如果替换前后的词没有相似性，可能是错误替换
            if removed_words and added_words:
                for removed in removed_words:
                    for added in added_words:
                        # 如果原词和新词完全不同（编辑距离过大），判定为错误替换
                        if len(removed) > 3 and len(added) > 3 and not any(c in removed for c in added):
                            return True
            return False

        # 统计变更类型
        format_changes = 0
        content_changes = 0
        wrong_changes = 0

        # 遍历所有日志
        for log in batch_logs:
            if log["type"] == "content_optimization":
                original = log["original"]
                optimized = log["optimized"]

                # 分类统计
                if is_format_change_only(original, optimized):
                    format_changes += 1
                elif is_wrong_replacement(original, optimized):
                    wrong_changes += 1
                else:
                    content_changes += 1

        total_changes = format_changes + content_changes + wrong_changes

        return {
            'format_changes': format_changes,
            'content_changes': content_changes,
            'wrong_changes': wrong_changes,
            'total_changes': total_changes
        }

    def _format_time_stats(self, stages: dict, total_time: float) -> None:
        """格式化显示时间统计"""
        print(f"⏱️  [bold blue]耗时统计:[/bold blue]")

        # 检查是否有并行处理阶段
        has_parallel = "⚡ 并行预处理" in stages

        if has_parallel:
            print(f"   📊 [bold yellow]并行优化效果:[/bold yellow]")

            # 计算并行处理的优化效果
            parallel_time = stages.get("⚡ 并行预处理", 0)
            split_time = stages.get("✂️  智能断句", 0)
            summary_time = stages.get("🔍 内容分析", 0)

            if split_time > 0 and summary_time > 0:
                serial_time = split_time + summary_time  # 串行处理需要的时间
                time_saved = serial_time - parallel_time  # 节省的时间
                efficiency_gain = (time_saved / serial_time) * 100 if serial_time > 0 else 0

                print(f"      ⚡ 并行处理: [cyan]{parallel_time:.1f}s[/cyan]")
                print(f"      📏 断句时间: [dim]{split_time:.1f}s[/dim]")
                print(f"      🔍 分析时间: [dim]{summary_time:.1f}s[/dim]")
                print(f"      ⏱️  串行耗时: [dim]{serial_time:.1f}s[/dim]")
                print(f"      💡 节省时间: [green]{time_saved:.1f}s[/green] ([green]{efficiency_gain:.0f}%[/green])")
                print()

        # 按执行顺序显示各阶段（保持字典插入顺序）
        for stage_name, elapsed_time in stages.items():
            if elapsed_time > 0 and stage_name != "⚡ 并行预处理":  # 并行处理已单独显示
                percentage = (elapsed_time / total_time) * 100
                print(f"   {stage_name}: [cyan]{elapsed_time:.1f}s[/cyan] ([dim]{percentage:.0f}%[/dim])")

        print(f"   [bold]总计: [cyan]{total_time:.1f}s[/cyan][/bold]") 