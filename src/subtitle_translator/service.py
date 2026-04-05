"""
字幕翻译服务模块 - 核心翻译服务类
"""
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from rich import print

from .exceptions import OpenAIAPIError, EmptySubtitleError, TranslationError, SmartSplitError
from .logger import log_section_end, log_section_start, log_stats
from .translation_core.config import SubtitleConfig
from .translation_core.context_extractor import extract_context_info
from .translation_core.data import SubtitleData, load_subtitle
from .translation_core.optimizer import SubtitleOptimizer, format_diff, _is_format_change_only, _is_wrong_replacement
from .translation_core.splitter import (
    batch_by_sentence_count,
    merge_segments_within_batch,
    preprocess_segments,
    presplit_by_punctuation,
)
from .translation_core.terminology import load_terminology
from .context_loader import build_context_info
from .console_views import show_api_config, show_model_config, show_time_stats


class SubtitleTranslatorService:
    """字幕翻译服务类"""
    
    def __init__(self):
        self.config = SubtitleConfig()
        from .env_setup import logger
        self.logger = logger

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
            show_model_config(self.config.split_model, self.config.translation_model)

        elapsed_time = time.time() - start_time
        log_section_end(self.logger, "翻译环境初始化", elapsed_time, "✅")

    def unload_local_models_if_needed(self) -> None:
        """在批量任务完成后主动卸载本次使用的本地 LM Studio 模型。"""
        if not self.config.lm_studio_unload_on_complete:
            return

        parsed = urlparse(self.config.openai_base_url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            self.logger.info("跳过主动卸载：当前 OpenAI-compatible 服务不是本机 LM Studio")
            return

        model_ids = {
            model_name
            for model_name in (self.config.split_model, self.config.translation_model)
            if model_name
        }
        if not model_ids:
            return

        for model_id in sorted(model_ids):
            result = subprocess.run(
                ["lms", "unload", model_id],
                capture_output=True,
                text=True,
                check=False,
            )
            output = (result.stdout or result.stderr).strip()
            if result.returncode == 0:
                self.logger.info(f"🧹 已主动卸载模型: {model_id}")
            else:
                self.logger.warning(f"主动卸载模型失败: {model_id} {output}")

    def _save_subtitle_files(
        self,
        asr_data: SubtitleData,
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

        asr_data.save_translations_to_files(
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

        asr_data = load_subtitle(str(input_srt_path))
        self.logger.info(f"📊 字幕统计: 共 {len(asr_data.segments)} 条字幕")
        self.logger.info(f"字幕内容预览: {asr_data.to_txt()[:100]}...")

        if len(asr_data.segments) == 0:
            self.logger.info("⚠️  SRT文件为空，跳过翻译处理")
            print("[yellow]⚠️  SRT文件为空，跳过翻译处理[/yellow]")
            raise EmptySubtitleError("SRT文件为空，无法进行翻译")

        print("📊 [bold blue]加载完成[/bold blue]")
        return asr_data

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
                for term, translation in self.config.terminology.items():
                    self.logger.info(f"   {term} → {translation}")
            else:
                self.logger.info("📚 未加载任何术语表")

            # 只在需要时初始化翻译环境
            if not skip_env_init:
                self.init_translation_env(llm_model)

            # 加载字幕文件
            asr_data = self._load_subtitle_file(input_srt_path)

            preprocessing_start_time = time.time()
            log_section_start(self.logger, "并行预处理阶段", "⚡")
            print("⚡ [bold cyan]启动并行预处理：上下文提炼 + 断句准备...[/bold cyan]")

            with ThreadPoolExecutor(max_workers=2) as executor:
                context_future = executor.submit(self._extract_translation_context, input_srt_path.resolve(), asr_data)
                prepare_future = executor.submit(self._prepare_subtitles_for_translation, asr_data)

                context_info, context_time = context_future.result()
                asr_data, split_time = prepare_future.result()

            stage_times["📋 上下文提炼"] = context_time
            stage_times["✂️ 断句准备"] = split_time

            self._log_context_info(context_info)

            preprocessing_time = time.time() - preprocessing_start_time
            log_section_end(self.logger, "并行预处理阶段", preprocessing_time, "🎉")
            print(f"🎉 [bold green]并行预处理完成[/bold green] (总耗时: [cyan]{preprocessing_time:.1f}s[/cyan])")
            stage_times["⚡ 并行预处理"] = preprocessing_time

            translate_start_time = time.time()
            print("🌍 [bold cyan]开始统一翻译...[/bold cyan]")
            translate_result = self._translate_prepared_subtitles(asr_data, context_info)
            stage_times["🌍 统一翻译"] = time.time() - translate_start_time

            target_lang_output_path = self._save_subtitle_files(
                asr_data, translate_result, input_srt_path, output_dir, target_lang
            )

            total_elapsed = time.time() - task_start_time

            print()
            show_time_stats(stage_times, total_elapsed)

            final_stats = {
                "输入文件": input_srt_path.name,
                "字幕数量": len(asr_data.segments),
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

    def _extract_translation_context(self, input_srt_path: Path, asr_data: SubtitleData) -> Tuple[str, float]:
        """提炼翻译上下文，失败时回退到本地元数据。"""
        start_time = time.time()
        try:
            context_info = extract_context_info(input_srt_path, asr_data, self.config)
            return context_info, time.time() - start_time
        except Exception as exc:
            self.logger.warning(f"⚠️ LLM上下文提炼失败，回退到本地元数据: {exc}")
            return build_context_info(input_srt_path), time.time() - start_time

    def _log_context_info(self, context_info: str) -> None:
        """打印加载的上下文信息。"""
        if context_info:
            self.logger.info("📋 已加载上下文信息:")
            for line in context_info.split('\n'):
                if line.strip():
                    self.logger.info(f"   {line}")
        else:
            self.logger.info("📋 未加载任何上下文信息")

    def _prepare_subtitles_for_translation(self, asr_data: SubtitleData) -> Tuple[SubtitleData, float]:
        """完成断句准备阶段，返回可直接翻译的字幕数据。"""
        start_time = time.time()

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
        self.logger.info(f"📦 分为 {total_batches} 批处理 {len(word_segments)} 个单词")

        # 5. 并发断句
        concurrency = self.config.thread_num
        all_segments = []

        def process_batch_task(args):
            """每个批次的断句任务。"""
            batch_index, batch = args
            return merge_segments_within_batch(
                batch, word_segments,
                model=self.config.split_model,
                batch_index=batch_index + 1
            )

        batch_tasks = list(enumerate(batches))

        for i in range(0, len(batch_tasks), concurrency):
            chunk = batch_tasks[i:i + concurrency]
            with ThreadPoolExecutor(max_workers=min(len(chunk), concurrency)) as executor:
                processed_chunks = list(executor.map(process_batch_task, chunk))
                for segments in processed_chunks:
                    all_segments.extend(segments)

                progress = min(i + concurrency, len(batch_tasks))
                self.logger.info(f"📈 断句进度: {progress}/{len(batch_tasks)}")

        # 6. 按时间排序
        all_segments.sort(key=lambda seg: seg.start_time)
        final_asr_data = SubtitleData(all_segments)

        self.logger.info(f"✅ 断句准备完成！共 {len(all_segments)} 句")
        return final_asr_data, time.time() - start_time

    def _translate_prepared_subtitles(self, asr_data: SubtitleData, context_info: str) -> list:
        """在上下文和断句都准备完毕后统一翻译。"""
        translator = SubtitleOptimizer(config=self.config)
        all_translated_results = translator.translate_subtitles(asr_data, context_info)
        batch_logs_all = translator.batch_logs

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
            print("📊 [bold blue]优化统计:[/bold blue]")
            if stats['format_changes'] > 0:
                print(f"   格式优化: [cyan]{stats['format_changes']}[/cyan] 项")
            if stats['content_changes'] > 0:
                print(f"   内容修改: [cyan]{stats['content_changes']}[/cyan] 项")
            if stats['wrong_changes'] > 0:
                print(f"   [yellow]可疑替换: {stats['wrong_changes']} 项[/yellow]")
            print(f"   总计: [cyan]{stats['total_changes']}[/cyan] 项优化")

        self.logger.info(f"✅ 统一翻译完成！共 {len(asr_data.segments)} 句")
        return renumbered_results

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
