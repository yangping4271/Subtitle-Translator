"""
字幕翻译服务模块 - 核心翻译服务类
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from rich import print

from .exceptions import OpenAIAPIError, EmptySubtitleError, TranslationError, SmartSplitError
from .logger import log_section_end, log_section_start, log_stats, setup_logger
from .translation_core.config import SubtitleConfig
from .translation_core.data import SubtitleData, load_subtitle
from .translation_core.external_glossary import load_external_terminology
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
    load_terminology,
)
from .translation_core.splitter import SubtitleSegmenter
from .context_loader import build_context_info
from .console_views import show_api_config, show_model_config, show_time_stats


class SubtitleTranslatorService:
    """字幕翻译服务类"""
    
    def __init__(
        self,
        config: Optional[SubtitleConfig] = None,
        llm: Optional[ModelAdapter] = None,
    ):
        self.config = config or SubtitleConfig()
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
        split_model: Optional[str] = None,
        translation_model: Optional[str] = None,
        show_config: bool = True
    ) -> None:
        """初始化翻译环境配置"""
        start_time = time.time()
        log_section_start(self.logger, "翻译环境初始化", "⚙️")

        if llm_model:
            self.config.split_model = llm_model
            self.config.translation_model = llm_model

        if split_model:
            self.config.split_model = split_model
        if translation_model:
            self.config.translation_model = translation_model

        self.logger.info(f"🌐 API端点: {self.config.openai_base_url}")

        model_config = {
            "断句模型": self.config.split_model,
            "翻译模型": self.config.translation_model
        }
        log_stats(self.logger, model_config, "模型配置")

        if show_config:
            show_api_config(self.config.openai_base_url, self.config.openai_api_key)
            show_model_config(
                self.config.split_model,
                self.config.translation_model,
                provider_type=self.config.provider_type(),
                disable_thinking=self.config.disable_thinking,
            )

        elapsed_time = time.time() - start_time
        log_section_end(self.logger, "翻译环境初始化", elapsed_time, "✅")

    def _save_subtitle_files(
        self,
        sentence_subtitle: SubtitleData,
        translate_result: list,
        input_srt_path: Path,
        output_dir: Path,
        target_lang: str
    ) -> Path:
        """保存翻译结果到文件"""
        self.logger.info("💾 正在保存翻译结果...")

        base_name = input_srt_path.stem
        target_lang_output_path = output_dir / f"{base_name}.{target_lang}.srt"
        english_output_path = output_dir / f"{base_name}.en.srt"

        self.logger.info(f"翻译文件将保存到目录: {output_dir}")
        self.logger.info(f"目标语言文件: {target_lang_output_path}")
        self.logger.info(f"英文文件: {english_output_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        sentence_subtitle.save_translations_to_files(
            translate_result,
            str(english_output_path),
            str(target_lang_output_path)
        )

        if not target_lang_output_path.exists():
            raise RuntimeError(f"目标语言翻译文件保存失败: {target_lang_output_path}")
        if not english_output_path.exists():
            raise RuntimeError(f"英文翻译文件保存失败: {english_output_path}")

        self.logger.info("翻译文件已保存:")
        self.logger.info(f"  - 目标语言: {target_lang_output_path}")
        self.logger.info(f"  - 英文: {english_output_path}")

        return target_lang_output_path

    def _load_subtitle_file(self, input_srt_path: Path) -> SubtitleData:
        """加载并验证字幕文件"""
        self.logger.info("📂 正在加载字幕文件...")

        source_subtitle = load_subtitle(str(input_srt_path))
        timestamp_type = "词级时间戳" if source_subtitle.is_word_timestamp() else "句段时间戳"
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

    def _set_target_language(self, target_lang: str) -> None:
        """设置目标语言（带友好错误处理）"""
        self.logger.info(f"🌍 设置目标语言: {target_lang}")

        try:
            self.config.set_target_language(target_lang)
            self.logger.info(f"✅ 目标语言已设置为: {self.config.target_language}")
        except ValueError as e:
            self.logger.error(f"❌ 语言设置失败: {str(e)}")
            print("[bold red]❌ 语言设置失败![/bold red]")
            print(str(e))
            raise

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
        try:
            task_start_time = time.time()
            log_section_start(self.logger, "字幕翻译任务", "🎬")

            # 用于收集各阶段耗时的字典
            stage_times = {}

            # 设置目标语言
            self._set_target_language(target_lang)

            # 加载术语表（全局 + 局部）
            self.config.terminology = load_terminology(
                self.config.target_language,
                input_srt_path
            )

            # 打印加载的术语表
            if self.config.terminology:
                self.logger.info(f"📚 已加载术语表: {len(self.config.terminology)} 条术语")
                for term, entry in self.config.terminology.items():
                    translation = get_terminology_translation(entry)
                    aliases = get_terminology_aliases(entry)
                    alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
                    self.logger.info(f"   {term} → {translation}{alias_text}")
            else:
                self.logger.info("📚 未加载任何术语表")

            self.config.external_terminology = {}
            if self.config.external_glossary_enabled:
                self.config.external_terminology = load_external_terminology(
                    self.config.target_language,
                    self.config.external_glossary_domains,
                )
                if self.config.external_terminology:
                    domains = ", ".join(self.config.external_glossary_domains)
                    self.logger.info(
                        f"📚 已加载外部术语库: {len(self.config.external_terminology)} 条术语 "
                        f"(domains: {domains}, dynamic max: {self.config.external_glossary_max_terms})"
                    )
                else:
                    self.logger.info("📚 未加载外部术语库")

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
                for line in context_info.split('\n'):
                    if line.strip():
                        self.logger.info(f"   {line}")
            else:
                self.logger.info("📋 未加载任何上下文信息")

            print("⚡ [bold cyan]启动字幕处理：并发断句 → 并发翻译...[/bold cyan]")

            sentence_subtitle, translate_result, processing_times = (
                self._translate_segmented_batches(
                    source_subtitle,
                    context_info,
                )
            )
            stage_times.update(processing_times)

            processing_time = time.time() - processing_start_time
            log_section_end(self.logger, "字幕处理阶段", processing_time, "🎉")
            print(f"🎉 [bold green]字幕处理完成[/bold green] (总耗时: [cyan]{processing_time:.1f}s[/cyan])")

            save_start_time = time.time()
            target_lang_output_path = self._save_subtitle_files(
                sentence_subtitle,
                translate_result,
                input_srt_path,
                output_dir,
                target_lang,
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
                "总耗时": f"{total_elapsed:.1f}秒"
            }
            log_stats(self.logger, final_stats, "任务完成统计")
            log_section_end(self.logger, "字幕翻译任务", total_elapsed, "🎉")

            return target_lang_output_path

        except OpenAIAPIError as e:
            self.logger.error(f"🚨 API错误: {str(e)}")
            raise

        except Exception as e:
            if isinstance(e, (SmartSplitError, TranslationError, EmptySubtitleError)):
                raise e

            self.logger.error(f"💥 处理过程中发生错误: {str(e)}")
            self.logger.debug("详细错误信息:", exc_info=True)
            raise

    def _translate_segmented_batches(
        self,
        source_subtitle: SubtitleData,
        context_info: str,
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

        with TranslationEngine(self.config, self.llm) as translator:
            def process_batch_task(args):
                """翻译一个已完成 Subtitle segmentation 的 Translation batch。"""
                batch_index, translation_batch = args
                batch_translate_result = translator.translate_batch(
                    translation_batch,
                    context_info,
                    batch_num=batch_index + 1,
                    total_batches=total_batches,
                )

                return batch_index, list(translation_batch.segments), batch_translate_result

            batch_tasks = list(enumerate(batches))

            for i in range(0, len(batch_tasks), concurrency):
                chunk = batch_tasks[i:i + concurrency]
                with ThreadPoolExecutor(max_workers=min(len(chunk), concurrency)) as executor:
                    future_to_batch_index = {
                        executor.submit(process_batch_task, batch_task): batch_task[0]
                        for batch_task in chunk
                    }
                    chunk_results = {}

                    for future in as_completed(future_to_batch_index):
                        batch_index, segments, translate_result = future.result()
                        chunk_results[batch_index] = (segments, translate_result)
                        completed_batches += 1
                        self.logger.info(f"📈 翻译进度: {completed_batches}/{len(batch_tasks)}")
                        print(
                            "📈 [bold cyan]批次进度:[/bold cyan] "
                            f"[cyan]{completed_batches}/{len(batch_tasks)}[/cyan] "
                            f"(当前完成: 第 {batch_index + 1} 批)"
                        )

                    for batch_index in sorted(chunk_results):
                        segments, translate_result = chunk_results[batch_index]
                        all_segments.extend(segments)
                        all_translated_results.extend(translate_result)

            batch_logs_all = list(translator.batch_logs)
        translation_time = time.time() - translation_start

        # 6. 按时间排序
        all_segments.sort(key=lambda seg: seg.start_time)
        sentence_subtitle = SubtitleData(all_segments)

        # 7. 重新编号翻译结果
        renumbered_results = []
        for idx, result in enumerate(all_translated_results, 1):
            result_copy = result.copy()
            result_copy['id'] = idx
            renumbered_results.append(result_copy)

        # 8. 显示优化统计
        stats = self._get_optimization_stats(batch_logs_all)
        self.logger.info(
            "📊 优化统计: 格式=%s, 内容=%s, 可疑=%s, 总计=%s",
            stats["format_changes"],
            stats["content_changes"],
            stats["wrong_changes"],
            stats["total_changes"],
        )
        if stats['total_changes'] > 0:
            # 先显示详细的优化日志
            if self.config.log_raw_payloads:
                self._print_optimization_details(batch_logs_all)

            # 再显示汇总统计
            print("📊 [bold blue]优化统计:[/bold blue]")
            if stats['format_changes'] > 0:
                print(f"   格式优化: [cyan]{stats['format_changes']}[/cyan] 项")
            if stats['content_changes'] > 0:
                print(f"   内容修改: [cyan]{stats['content_changes']}[/cyan] 项")
            if stats['wrong_changes'] > 0:
                print(f"   [yellow]可疑替换: {stats['wrong_changes']} 项[/yellow]")
            print(f"   总计: [cyan]{stats['total_changes']}[/cyan] 项优化")

        self.logger.info(f"✅ 字幕处理完成！共 {len(all_segments)} 个句段")

        return sentence_subtitle, renumbered_results, {
            "✂️ 智能断句": segmentation_time,
            "🌍 批量翻译": translation_time,
        }

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
            'format_changes': format_changes,
            'content_changes': content_changes,
            'wrong_changes': wrong_changes,
            'total_changes': format_changes + content_changes + wrong_changes
        }
