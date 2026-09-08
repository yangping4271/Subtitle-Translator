"""
字幕翻译服务模块 - 核心翻译服务类
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from rich import print

from .exceptions import (
    OpenAIAPIError,
    EmptySubtitleError,
    TranslationError,
    SmartSplitError,
)
from .logger import log_section_end, log_section_start, log_stats, setup_logger
from .output_files import SubtitleOutputFiles, write_subtitle_outputs
from .translation_core.config import SubtitleConfig
from .translation_core.data import SubtitleData, load_subtitle
from .translation_core.llm_client import LLMClient, ModelAdapter
from .translation_core.translation_execution import (
    TranslationEngine,
    _is_format_change_only,
    _is_wrong_replacement,
    format_diff,
)
from .translation_core.terminology import (
    get_terminology_aliases,
    get_terminology_translation,
)
from .translation_core.translation_context import TranslationContext
from .translation_core.splitter import SubtitleSegmenter
from .context_loader import build_context_info
from .console_views import (
    show_api_config,
    show_api_performance_stats,
    show_model_config,
    show_time_stats,
)


class SubtitleTranslatorService:
    """字幕翻译服务类"""

    def __init__(
        self,
        config: Optional[SubtitleConfig] = None,
        llm: Optional[ModelAdapter] = None,
    ):
        self.config = config or SubtitleConfig.from_env()
        self._owns_llm = llm is None
        self.llm = llm or LLMClient(self.config)
        self.logger = setup_logger(__name__)

    def close(self) -> None:
        """释放由当前运行创建的 external adapter。"""
        if self._owns_llm and isinstance(self.llm, LLMClient):
            self.llm.close()
            self._owns_llm = False

    def init_translation_env(
        self,
        llm_model: Optional[str] = None,
        show_config: bool = True,
    ) -> None:
        """初始化翻译环境配置"""
        start_time = time.time()
        log_section_start(self.logger, "翻译环境初始化", "⚙️")

        if llm_model:
            self.config.llm_model = llm_model

        self.logger.info(f"🌐 API端点: {self.config.openai_base_url}")

        log_stats(self.logger, {"模型": self.config.llm_model}, "模型配置")

        if show_config:
            show_api_config(self.config.openai_base_url, self.config.openai_api_key)
            show_model_config(
                self.config.llm_model,
                provider_type=self.config.provider_type(),
                disable_thinking=self.config.disable_thinking,
            )

        elapsed_time = time.time() - start_time
        log_section_end(self.logger, "翻译环境初始化", elapsed_time, "✅")

    def _load_subtitle_file(self, input_srt_path: Path) -> SubtitleData:
        """加载并验证字幕文件"""
        self.logger.info("📂 正在加载字幕文件...")

        source_subtitle = load_subtitle(str(input_srt_path))
        timestamp_type = (
            "词级时间戳" if source_subtitle.is_word_timestamp() else "句段时间戳"
        )
        self.logger.info(
            "📊 输入字幕片段: %s 条（%s）",
            len(source_subtitle.segments),
            timestamp_type,
        )
        if self.config.log_raw_payloads:
            self.logger.debug(f"字幕内容预览: {source_subtitle.to_txt()[:100]}...")

        if len(source_subtitle.segments) == 0:
            self.logger.info("⚠️  SRT文件为空，跳过翻译处理")
            print("[yellow]⚠️  SRT文件为空，跳过翻译处理[/yellow]")
            raise EmptySubtitleError("SRT文件为空，无法进行翻译")

        print("📊 [bold blue]加载完成[/bold blue]")
        return source_subtitle

    def _load_translation_context(
        self,
        target_lang: str,
        input_srt_path: Path,
    ) -> TranslationContext:
        """加载单个 Source subtitle 的目标语言与术语。"""
        self.logger.info(f"🌍 设置目标语言: {target_lang}")
        try:
            translation_context = TranslationContext.for_file(
                self.config,
                target_lang,
                input_srt_path,
            )
        except ValueError as e:
            self.logger.error(f"❌ 语言设置失败: {str(e)}")
            print("[bold red]❌ 语言设置失败![/bold red]")
            print(str(e))
            raise

        self.logger.info(
            "✅ 目标语言已设置为: %s",
            translation_context.target_language,
        )
        if translation_context.terminology:
            self.logger.info(
                "📚 已加载术语表: %s 条术语",
                len(translation_context.terminology),
            )
            for term, entry in translation_context.terminology.items():
                translation = get_terminology_translation(entry)
                aliases = get_terminology_aliases(entry)
                alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
                self.logger.info(f"   {term} → {translation}{alias_text}")
        else:
            self.logger.info("📚 未加载任何术语表")

        if translation_context.external_terminology:
            domains = ", ".join(self.config.external_glossary_domains)
            self.logger.info(
                "📚 已加载外部术语库: %s 条术语 (domains: %s, dynamic max: %s)",
                len(translation_context.external_terminology),
                domains,
                self.config.external_glossary_max_terms,
            )
        elif self.config.external_glossary_enabled:
            self.logger.info("📚 未加载外部术语库")
        return translation_context

    def translate_srt(
        self,
        input_srt_path: Path,
        target_lang: str,
        output_dir: Path,
        llm_model: Optional[str] = None,
        skip_env_init: bool = False,
        preserve_intermediate: bool = True,
    ) -> SubtitleOutputFiles:
        """翻译字幕文件

        Args:
            input_srt_path: 输入字幕文件路径
            target_lang: 目标语言
            output_dir: 输出目录
            llm_model: LLM 模型名称
            skip_env_init: 是否跳过环境初始化
            preserve_intermediate: 是否保留英文和目标语言 SRT 中间文件
        """
        task_start_time = time.time()
        metrics_checkpoint = None
        try:
            if isinstance(self.llm, LLMClient):
                metrics_checkpoint = self.llm.metrics_checkpoint()
        except Exception:
            metrics_checkpoint = None
        try:
            log_section_start(self.logger, "字幕翻译任务", "🎬")

            # 用于收集各阶段耗时的字典
            stage_times = {}

            translation_context = self._load_translation_context(
                target_lang,
                input_srt_path,
            )

            # 只在需要时初始化翻译环境
            if not skip_env_init:
                self.init_translation_env(llm_model)

            # 加载字幕文件
            source_subtitle = self._load_subtitle_file(input_srt_path)

            processing_start_time = time.time()
            log_section_start(self.logger, "字幕处理阶段", "⚡")

            context_start_time = time.time()
            context_info = build_context_info(input_srt_path.resolve())
            stage_times["📋 上下文提取"] = time.time() - context_start_time

            # 打印加载的上下文信息
            if context_info:
                self.logger.info("📋 已加载上下文信息:")
                for line in context_info.split("\n"):
                    if line.strip():
                        self.logger.info(f"   {line}")
            else:
                self.logger.info("📋 未加载任何上下文信息")

            print("⚡ [bold cyan]启动字幕处理：并发断句 → 并发翻译...[/bold cyan]")

            sentence_subtitle, translate_result, processing_times = (
                self._translate_segmented_batches(
                    source_subtitle,
                    context_info,
                    translation_context,
                )
            )
            stage_times.update(processing_times)

            processing_time = time.time() - processing_start_time
            log_section_end(self.logger, "字幕处理阶段", processing_time, "🎉")
            print(
                f"🎉 [bold green]字幕处理完成[/bold green] (总耗时: [cyan]{processing_time:.1f}s[/cyan])"
            )

            save_start_time = time.time()
            self.logger.info("💾 正在生成字幕输出文件...")
            output_files = write_subtitle_outputs(
                sentence_subtitle=sentence_subtitle,
                translation_results=translate_result,
                input_srt_path=input_srt_path,
                output_dir=output_dir,
                target_lang=target_lang,
                preserve_intermediate=preserve_intermediate,
            )
            stage_times["💾 保存字幕"] = time.time() - save_start_time

            total_elapsed = time.time() - task_start_time

            print()
            show_time_stats(stage_times, total_elapsed)
            final_stats = {
                "输入文件": input_srt_path.name,
                "输入字幕片段": len(source_subtitle.segments),
                "断句后句段": len(sentence_subtitle.segments),
                "目标语言": target_lang,
                "总耗时": f"{total_elapsed:.1f}秒",
            }
            log_stats(self.logger, final_stats, "任务完成统计")
            log_section_end(self.logger, "字幕翻译任务", total_elapsed, "🎉")

            return output_files

        except OpenAIAPIError as e:
            self.logger.error(f"🚨 API错误: {str(e)}")
            raise

        except Exception as e:
            if isinstance(e, (SmartSplitError, TranslationError, EmptySubtitleError)):
                raise e

            self.logger.error(f"💥 处理过程中发生错误: {str(e)}")
            self.logger.debug("详细错误信息:", exc_info=True)
            raise
        finally:
            if metrics_checkpoint is not None:
                self._show_api_metrics(metrics_checkpoint)

    def _show_api_metrics(self, checkpoint: int) -> None:
        """输出 checkpoint 之后的请求指标，失败任务也保留汇总。"""
        try:
            api_stats = self.llm.metrics_summary(checkpoint)
            if api_stats["requests"] == 0:
                return
            print()
            show_api_performance_stats(api_stats)
            self.logger.info(
                "📡 API性能统计: 请求=%s, 成功=%s, 失败=%s, "
                "平均延迟=%s, P95=%s, 最大延迟=%s, 有效吞吐=%s, "
                "吞吐覆盖请求=%s, usage缺失=%s, "
                "慢请求=%s, 响应异常=%s",
                api_stats["requests"],
                api_stats["successful_requests"],
                api_stats["failed_requests"],
                (
                    f"{api_stats['latency_avg']:.2f}s"
                    if api_stats["latency_avg"] is not None
                    else "unknown"
                ),
                (
                    f"{api_stats['latency_p95']:.2f}s"
                    if api_stats["latency_p95"] is not None
                    else "unknown"
                ),
                (
                    f"{api_stats['latency_max']:.2f}s"
                    if api_stats["latency_max"] is not None
                    else "unknown"
                ),
                (
                    f"{api_stats['effective_tps']:.2f} token/s"
                    if api_stats["effective_tps"] is not None
                    else "unknown"
                ),
                api_stats["throughput_requests"],
                api_stats["missing_usage"],
                api_stats["slow_requests"],
                api_stats["anomalies"],
            )
        except Exception as exc:
            try:
                self.logger.warning(
                    "API性能统计输出失败，已保留原翻译结果: error_type=%s",
                    type(exc).__name__,
                )
            except Exception:
                pass

    def _translate_segmented_batches(
        self,
        source_subtitle: SubtitleData,
        context_info: str,
        translation_context: TranslationContext,
    ) -> Tuple[SubtitleData, list, dict[str, float]]:
        """
        先完成 Subtitle segmentation，再并发执行 Translation batches。

        Returns:
            (sentence_subtitle, translation_results, stage_times)
        """
        segmentation_start = time.time()
        segmenter = SubtitleSegmenter(self.config, self.llm)
        batches = segmenter.segment(source_subtitle)
        segmentation_time = time.time() - segmentation_start
        total_batches = len(batches)
        segment_count = sum(len(batch.segments) for batch in batches)
        batch_sizes = [len(batch.segments) for batch in batches]
        self.logger.info(
            "📦 断句完成: %s 个句段，翻译批次=%s，批次大小=%s",
            segment_count,
            total_batches,
            batch_sizes,
        )
        print(f"📦 [bold cyan]批次总数:[/bold cyan] [cyan]{total_batches}[/cyan]")

        translation_start = time.time()
        concurrency = self.config.thread_num
        all_translated_results = []
        all_segments = []
        completed_batches = 0

        with TranslationEngine(
            self.config,
            self.llm,
            translation_context,
        ) as translator:

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_batch_index = {
                    executor.submit(
                        translator.translate_batch,
                        batch,
                        context_info,
                        batch_num=index + 1,
                        total_batches=total_batches,
                    ): index
                    for index, batch in enumerate(batches)
                }
                batch_results = {}
                for future in as_completed(future_to_batch_index):
                    batch_index = future_to_batch_index[future]
                    batch_results[batch_index] = future.result()
                    completed_batches += 1
                    self.logger.info(f"📈 翻译进度: {completed_batches}/{total_batches}")
                    print(
                        "📈 [bold cyan]批次进度:[/bold cyan] "
                        f"[cyan]{completed_batches}/{total_batches}[/cyan] "
                        f"(当前完成: 第 {batch_index + 1} 批)"
                    )

                for batch_index, batch in enumerate(batches):
                    all_segments.extend(batch.segments)
                    all_translated_results.extend(batch_results[batch_index])

            batch_logs_all = list(translator.batch_logs)
        translation_time = time.time() - translation_start

        sentence_subtitle = SubtitleData(all_segments)

        renumbered_results = [
            {**result, "id": index}
            for index, result in enumerate(all_translated_results, 1)
        ]

        # 显示优化统计
        stats = self._get_optimization_stats(batch_logs_all)
        self.logger.info(
            "📊 优化统计: 格式=%s, 内容=%s, 可疑=%s, 总计=%s",
            stats["format_changes"],
            stats["content_changes"],
            stats["wrong_changes"],
            stats["total_changes"],
        )
        if stats["total_changes"] > 0:
            # 先显示详细的优化日志
            if self.config.log_raw_payloads:
                self._print_optimization_details(batch_logs_all)

            # 再显示汇总统计
            print("📊 [bold blue]优化统计:[/bold blue]")
            if stats["format_changes"] > 0:
                print(f"   格式优化: [cyan]{stats['format_changes']}[/cyan] 项")
            if stats["content_changes"] > 0:
                print(f"   内容修改: [cyan]{stats['content_changes']}[/cyan] 项")
            if stats["wrong_changes"] > 0:
                print(f"   [yellow]可疑替换: {stats['wrong_changes']} 项[/yellow]")
            print(f"   总计: [cyan]{stats['total_changes']}[/cyan] 项优化")

        self.logger.info(f"✅ 字幕处理完成！共 {len(all_segments)} 个句段")

        return (
            sentence_subtitle,
            renumbered_results,
            {
                "✂️ 智能断句": segmentation_time,
                "🌍 批量翻译": translation_time,
            },
        )

    def _print_optimization_details(self, batch_logs: list) -> None:
        """打印详细的优化日志"""
        self.logger.info("📊 字幕优化结果汇总")

        # 遍历所有日志，打印有实际改动的
        for log in batch_logs:
            if log["type"] == "content_optimization":
                id_num = log["id"]
                original = log["original"]
                optimized = log["optimized"]

                # 只在实际有变化时打印
                if original != optimized:
                    self.logger.info(f"🔧 字幕ID {id_num} - 内容优化:")
                    self.logger.info(f"   {format_diff(original, optimized)}")

    def _get_optimization_stats(self, batch_logs: list) -> dict:
        """从batch_logs中获取优化统计信息"""
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
            "format_changes": format_changes,
            "content_changes": content_changes,
            "wrong_changes": wrong_changes,
            "total_changes": format_changes + content_changes + wrong_changes,
        }
