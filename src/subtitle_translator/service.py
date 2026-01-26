"""
字幕翻译服务模块 - 核心翻译服务类
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

from rich import print

from .exceptions import OpenAIAPIError, EmptySubtitleError, TranslationError, SummaryError, SmartSplitError
from .logger import log_section_end, log_section_start, log_stats
from .translation_core.config import SubtitleConfig
from .translation_core.data import SubtitleData
from .translation_core.optimizer import SubtitleOptimizer
from .translation_core.spliter import (
    batch_by_sentence_count,
    merge_segments_within_batch,
    preprocess_segments,
    presplit_by_punctuation,
)
from .translation_core.summarizer import SubtitleSummarizer


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

    def _init_translation_env(
        self,
        llm_model: Optional[str] = None,
        split_model: Optional[str] = None,
        summary_model: Optional[str] = None,
        translation_model: Optional[str] = None,
        show_config: bool = True
    ) -> None:
        """初始化翻译环境配置

        Args:
            llm_model: 覆盖所有模型（优先级低于独立参数）
            split_model: 断句模型（优先级最高）
            summary_model: 总结模型（优先级最高）
            translation_model: 翻译模型（优先级最高）
            show_config: 是否显示配置信息
        """
        logger = self._get_logger()
        start_time = time.time()
        log_section_start(logger, "翻译环境初始化", "⚙️")

        # 优先级：独立参数 > llm_model > 环境变量 > 默认值
        if llm_model:
            self.config.split_model = llm_model
            self.config.summary_model = llm_model
            self.config.translation_model = llm_model

        # 独立参数覆盖（优先级最高）
        if split_model:
            self.config.split_model = split_model
        if summary_model:
            self.config.summary_model = summary_model
        if translation_model:
            self.config.translation_model = translation_model

        logger.info(f"🌐 API端点: {self.config.openai_base_url}")

        model_config = {
            "断句模型": self.config.split_model,
            "总结模型": self.config.summary_model,
            "翻译模型": self.config.translation_model
        }
        log_stats(logger, model_config, "模型配置")

        if show_config:
            self._display_api_config()
            self._display_model_config()

        elapsed_time = time.time() - start_time
        log_section_end(logger, "翻译环境初始化", elapsed_time, "✅")

    def _save_subtitle_files(
        self,
        asr_data: SubtitleData,
        translate_result: list,
        input_srt_path: Path,
        output_dir: Path,
        target_lang: str
    ) -> Path:
        """保存翻译结果到文件"""
        logger = self._get_logger()
        logger.info("💾 正在保存翻译结果...")

        base_name = input_srt_path.stem
        target_lang_output_path = output_dir / f"{base_name}.{target_lang}.srt"
        english_output_path = output_dir / f"{base_name}.en.srt"

        logger.info(f"翻译文件将保存到目录: {output_dir}")
        logger.info(f"目标语言文件: {target_lang_output_path}")
        logger.info(f"英文文件: {english_output_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        asr_data.save_translations_to_files(
            translate_result,
            str(english_output_path),
            str(target_lang_output_path)
        )

        if not target_lang_output_path.exists():
            raise RuntimeError(f"目标语言翻译文件保存失败: {target_lang_output_path}")
        if not english_output_path.exists():
            raise RuntimeError(f"英文翻译文件保存失败: {english_output_path}")

        logger.info(f"翻译文件已保存:")
        logger.info(f"  - 目标语言: {target_lang_output_path}")
        logger.info(f"  - 英文: {english_output_path}")

        return target_lang_output_path

    def _load_subtitle_file(self, input_srt_path: Path) -> SubtitleData:
        """加载并验证字幕文件"""
        from .translation_core.data import load_subtitle

        logger = self._get_logger()
        logger.info("📂 正在加载字幕文件...")

        asr_data = load_subtitle(str(input_srt_path))
        logger.info(f"📊 字幕统计: 共 {len(asr_data.segments)} 条字幕")
        logger.info(f"字幕内容预览: {asr_data.to_txt()[:100]}...")

        if len(asr_data.segments) == 0:
            logger.info("⚠️  SRT文件为空，跳过翻译处理")
            print(f"[yellow]⚠️  SRT文件为空，跳过翻译处理[/yellow]")
            raise EmptySubtitleError("SRT文件为空，无法进行翻译")

        print(f"📊 [bold blue]加载完成[/bold blue]")
        return asr_data

    def _set_target_language(self, target_lang: str) -> None:
        """设置目标语言（带友好错误处理）"""
        logger = self._get_logger()
        logger.info(f"🌍 设置目标语言: {target_lang}")

        try:
            self.config.set_target_language(target_lang)
            logger.info(f"✅ 目标语言已设置为: {self.config.target_language}")
        except ValueError as e:
            logger.error(f"❌ 语言设置失败: {str(e)}")
            print(f"[bold red]❌ 语言设置失败![/bold red]")
            print(str(e))
            raise

    def _display_api_config(self) -> None:
        """显示 API 配置信息"""
        print(f"🌐 [bold blue]API 配置:[/bold blue]")
        print(f"   端点: [cyan]{self.config.openai_base_url}[/cyan]")

        api_key = self.config.openai_api_key
        if api_key:
            masked_key = self._mask_api_key(api_key)
            print(f"   密钥: [cyan]{masked_key}[/cyan]")
        else:
            print(f"   密钥: [red]未设置[/red]")

    def _mask_api_key(self, api_key: str) -> str:
        """对 API 密钥进行脱敏处理"""
        if len(api_key) > 12:
            return f"{api_key[:6]}{'*' * 8}{api_key[-6:]}"
        return '*' * len(api_key)

    def _display_model_config(self) -> None:
        """显示模型配置信息"""
        print(f"🤖 [bold blue]模型配置:[/bold blue]")
        print(f"   断句: [cyan]{self.config.split_model}[/cyan]")
        print(f"   总结: [cyan]{self.config.summary_model}[/cyan]")
        print(f"   翻译: [cyan]{self.config.translation_model}[/cyan]")

    def translate_srt(self, input_srt_path: Path, target_lang: str, output_dir: Path,
                      llm_model: Optional[str] = None, skip_env_init: bool = False) -> Path:
        """翻译字幕文件

        Args:
            input_srt_path: 输入字幕文件路径
            target_lang: 目标语言
            output_dir: 输出目录
            llm_model: LLM 模型名称
            skip_env_init: 是否跳过环境初始化
        """
        logger = self._get_logger()
        try:
            task_start_time = time.time()
            log_section_start(logger, "字幕翻译任务", "🎬")
            
            # 用于收集各阶段耗时的字典
            stage_times = {}
            
            # 设置目标语言
            self._set_target_language(target_lang)
            
            # 只在需要时初始化翻译环境
            if not skip_env_init:
                self._init_translation_env(llm_model)
            
            # 加载字幕文件
            asr_data = self._load_subtitle_file(input_srt_path)
            
            # 并行预处理阶段：断句和总结同时进行（v0.5.x 性能优化）
            # 借鉴VideoCaptioner的解决方案：统一转换为单词级别后进行断句
            # 优势：1) 复用现有批量框架 2) 无额外API成本 3) 时间戳精确分配 4) 并行处理节省时间
            preprocessing_start_time = time.time()
            log_section_start(logger, "并行预处理阶段", "⚡")

            print(f"⚡ [bold cyan]启动并行预处理：断句 + 内容分析...[/bold cyan]")

            # 准备原始字幕内容用于总结（断句前）
            original_subtitle_content = asr_data.to_txt()

            # 启动总结任务（与流水线并行）
            def execute_summarization(subtitle_content: str, input_file: str) -> Tuple[dict, float]:
                """执行总结处理的任务函数"""
                summary_start_time = time.time()
                summarize_result = self._get_subtitle_summary(subtitle_content, input_file, is_parallel=True)
                summary_time = time.time() - summary_start_time
                return summarize_result, summary_time

            # 先获取总结（需要作为翻译上下文）
            summarize_result, summary_time = execute_summarization(original_subtitle_content, str(input_srt_path.resolve()))
            stage_times["🔍 内容分析"] = summary_time

            # 使用流水线式处理：断句 + 翻译一体化
            pipeline_start_time = time.time()
            print(f"⚡ [bold cyan]启动流水线处理：断句 + 翻译并行...[/bold cyan]")

            asr_data, translate_result = self._translate_with_pipeline(asr_data, summarize_result)

            pipeline_time = time.time() - pipeline_start_time
            stage_times["🚀 流水线处理"] = pipeline_time

            preprocessing_time = time.time() - preprocessing_start_time
            log_section_end(logger, "并行预处理阶段", preprocessing_time, "🎉")
            print(f"🎉 [bold green]流水线处理完成[/bold green] (总耗时: [cyan]{preprocessing_time:.1f}s[/cyan])")

            # 添加并行处理统计
            stage_times["⚡ 并行预处理"] = preprocessing_time
            
            # 保存字幕
            target_lang_output_path = self._save_subtitle_files(
                asr_data, translate_result, input_srt_path, output_dir, target_lang
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

    def _translate_with_pipeline(self, asr_data: SubtitleData, summarize_result: dict) -> Tuple[SubtitleData, list]:
        """
        流水线式翻译：每个批次断句后立即翻译

        Returns:
            (final_asr_data, translate_result)
        """
        logger = self._get_logger()

        # 1. 预处理：移除纯标点符号
        asr_data.segments = preprocess_segments(asr_data.segments)

        # 2. 转换为单词级字幕（如果需要）
        if not asr_data.is_word_timestamp():
            asr_data = asr_data.split_to_word_segments()

        word_segments = asr_data.segments

        # 3. 预分句
        pre_split_sentences = presplit_by_punctuation(word_segments)

        # 4. 分批
        batches = batch_by_sentence_count(
            pre_split_sentences,
            min_size=self.config.min_batch_sentences,
            max_size=self.config.max_batch_sentences
        )
        total_batches = len(batches)
        logger.info(f"📦 分为 {total_batches} 批处理 {len(word_segments)} 个单词")

        # 5. 并发处理
        concurrency = self.config.thread_num
        all_translated_results = []
        all_segments = []
        batch_logs_all = []

        def process_batch_task(args):
            """每个批次的完整任务：断句 + 翻译"""
            batch_index, batch = args
            batch_num = batch_index + 1

            batch_segments = merge_segments_within_batch(
                batch,
                word_segments,
                model=self.config.split_model,
                batch_index=batch_num
            )

            batch_asr_data = SubtitleData(batch_segments)
            translator = SubtitleOptimizer(config=self.config)
            batch_translate_result = translator.translate_batch_directly(batch_asr_data, summarize_result)

            return (batch_segments, batch_translate_result, translator.batch_logs)

        batch_tasks = list(enumerate(batches))

        for i in range(0, len(batch_tasks), concurrency):
            chunk = batch_tasks[i:i + concurrency]
            with ThreadPoolExecutor(max_workers=min(len(chunk), concurrency)) as executor:
                processed_chunks = list(executor.map(process_batch_task, chunk))
                for segments, translate_result, batch_logs in processed_chunks:
                    all_segments.extend(segments)
                    all_translated_results.extend(translate_result)
                    batch_logs_all.extend(batch_logs)

                progress = min(i + concurrency, len(batch_tasks))
                logger.info(f"📈 流水线进度: {progress}/{len(batch_tasks)}")

        # 6. 按时间排序
        all_segments.sort(key=lambda seg: seg.start_time)
        final_asr_data = SubtitleData(all_segments)

        # 7. 重新编号翻译结果
        renumbered_results = []
        for idx, result in enumerate(all_translated_results, 1):
            result_copy = result.copy()
            result_copy['id'] = idx
            renumbered_results.append(result_copy)

        # 8. 显示优化统计
        stats = self._get_optimization_stats(batch_logs_all)
        if stats['total_changes'] > 0:
            # 先显示详细的优化日志
            self._print_optimization_details(batch_logs_all)

            # 再显示汇总统计
            print(f"📊 [bold blue]优化统计:[/bold blue]")
            if stats['format_changes'] > 0:
                print(f"   格式优化: [cyan]{stats['format_changes']}[/cyan] 项")
            if stats['content_changes'] > 0:
                print(f"   内容修改: [cyan]{stats['content_changes']}[/cyan] 项")
            if stats['wrong_changes'] > 0:
                print(f"   [yellow]可疑替换: {stats['wrong_changes']} 项[/yellow]")
            print(f"   总计: [cyan]{stats['total_changes']}[/cyan] 项优化")

        logger.info(f"✅ 流水线处理完成！共 {len(all_segments)} 句")

        return final_asr_data, renumbered_results

    def _print_optimization_details(self, batch_logs: list) -> None:
        """打印详细的优化日志"""
        from .translation_core.optimizer import format_diff

        logger = self._get_logger()
        logger.info("📊 字幕优化结果汇总")

        # 遍历所有日志，打印有实际改动的
        for log in batch_logs:
            if log["type"] == "content_optimization":
                id_num = log["id"]
                original = log["original"]
                optimized = log["optimized"]

                # 只在实际有变化时打印
                if original != optimized:
                    logger.info(f"🔧 字幕ID {id_num} - 内容优化:")
                    logger.info(f"   {format_diff(original, optimized)}")

    def _get_optimization_stats(self, batch_logs: list) -> dict:
        """从batch_logs中获取优化统计信息"""
        from .translation_core.optimizer import _is_format_change_only, _is_wrong_replacement

        format_changes = 0
        content_changes = 0
        wrong_changes = 0

        for log in batch_logs:
            if log["type"] == "content_optimization":
                original = log["original"]
                optimized = log["optimized"]

                if _is_format_change_only(original, optimized):
                    format_changes += 1
                elif _is_wrong_replacement(original, optimized):
                    wrong_changes += 1
                else:
                    content_changes += 1

        return {
            'format_changes': format_changes,
            'content_changes': content_changes,
            'wrong_changes': wrong_changes,
            'total_changes': format_changes + content_changes + wrong_changes
        }

    def _format_time_stats(self, stages: dict, total_time: float) -> None:
        """格式化显示时间统计"""
        print(f"⏱️  [bold blue]耗时统计:[/bold blue]")

        # 检查是否有并行处理阶段
        has_parallel = "⚡ 并行预处理" in stages

        # 按执行顺序显示各阶段（保持字典插入顺序）
        for stage_name, elapsed_time in stages.items():
            if elapsed_time > 0 and stage_name != "⚡ 并行预处理":  # 并行处理不单独显示
                percentage = (elapsed_time / total_time) * 100
                print(f"   {stage_name}: [cyan]{elapsed_time:.1f}s[/cyan] ([dim]{percentage:.0f}%[/dim])")

        print(f"   [bold]总计: [cyan]{total_time:.1f}s[/cyan][/bold]") 