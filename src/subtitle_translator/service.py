"""
字幕翻译服务模块 - 核心翻译服务类
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

from rich import print

from .exceptions import OpenAIAPIError, EmptySubtitleError, TranslationError, SmartSplitError
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


class SubtitleTranslatorService:
    """字幕翻译服务类"""
    
    def __init__(self):
        self.config = SubtitleConfig()
        from .env_setup import logger
        self.logger = logger

    def _init_translation_env(
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
            self._display_api_config()
            self._display_model_config()

        elapsed_time = time.time() - start_time
        log_section_end(self.logger, "翻译环境初始化", elapsed_time, "✅")

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

        self.logger.info(f"翻译文件已保存:")
        self.logger.info(f"  - 目标语言: {target_lang_output_path}")
        self.logger.info(f"  - 英文: {english_output_path}")

        return target_lang_output_path

    def _load_subtitle_file(self, input_srt_path: Path) -> SubtitleData:
        """加载并验证字幕文件"""
        from .translation_core.data import load_subtitle

        self.logger.info("📂 正在加载字幕文件...")

        asr_data = load_subtitle(str(input_srt_path))
        self.logger.info(f"📊 字幕统计: 共 {len(asr_data.segments)} 条字幕")
        self.logger.info(f"字幕内容预览: {asr_data.to_txt()[:100]}...")

        if len(asr_data.segments) == 0:
            self.logger.info("⚠️  SRT文件为空，跳过翻译处理")
            print(f"[yellow]⚠️  SRT文件为空，跳过翻译处理[/yellow]")
            raise EmptySubtitleError("SRT文件为空，无法进行翻译")

        print(f"📊 [bold blue]加载完成[/bold blue]")
        return asr_data

    def _set_target_language(self, target_lang: str) -> None:
        """设置目标语言（带友好错误处理）"""
        self.logger.info(f"🌍 设置目标语言: {target_lang}")

        try:
            self.config.set_target_language(target_lang)
            self.logger.info(f"✅ 目标语言已设置为: {self.config.target_language}")
        except ValueError as e:
            self.logger.error(f"❌ 语言设置失败: {str(e)}")
            print(f"[bold red]❌ 语言设置失败![/bold red]")
            print(str(e))
            raise

    def _display_api_config(self) -> None:
        """显示 API 配置信息"""
        print(f"🌐 [bold blue]API 配置:[/bold blue]")
        print(f"   端点: [cyan]{self.config.openai_base_url}[/cyan]")

        api_key = self.config.openai_api_key
        masked_key = f"{api_key[:6]}{'*' * 8}{api_key[-6:]}" if len(api_key) > 12 else '*' * len(api_key)
        print(f"   密钥: [cyan]{masked_key}[/cyan]" if api_key else "   密钥: [red]未设置[/red]")

    def _display_model_config(self) -> None:
        """显示模型配置信息"""
        print(f"🤖 [bold blue]模型配置:[/bold blue]")
        print(f"   断句: [cyan]{self.config.split_model}[/cyan]")
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
        try:
            task_start_time = time.time()
            log_section_start(self.logger, "字幕翻译任务", "🎬")

            # 用于收集各阶段耗时的字典
            stage_times = {}

            # 设置目标语言
            self._set_target_language(target_lang)

            # 加载术语表（全局 + 局部）
            from .translation_core.terminology import load_terminology
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
                self._init_translation_env(llm_model)

            # 加载字幕文件
            asr_data = self._load_subtitle_file(input_srt_path)

            preprocessing_start_time = time.time()
            log_section_start(self.logger, "并行预处理阶段", "⚡")

            context_info = self._extract_context_info(str(input_srt_path.resolve()))
            stage_times["📋 上下文提取"] = 0.0  # 本地操作，耗时可忽略

            # 打印加载的上下文信息
            if context_info:
                self.logger.info("📋 已加载上下文信息:")
                for line in context_info.split('\n'):
                    if line.strip():
                        self.logger.info(f"   {line}")
            else:
                self.logger.info("📋 未加载任何上下文信息")

            pipeline_start_time = time.time()
            print(f"⚡ [bold cyan]启动流水线处理：断句 + 翻译并行...[/bold cyan]")

            asr_data, translate_result = self._translate_with_pipeline(asr_data, context_info)

            pipeline_time = time.time() - pipeline_start_time
            stage_times["🚀 流水线处理"] = pipeline_time

            preprocessing_time = time.time() - preprocessing_start_time
            log_section_end(self.logger, "并行预处理阶段", preprocessing_time, "🎉")
            print(f"🎉 [bold green]流水线处理完成[/bold green] (总耗时: [cyan]{preprocessing_time:.1f}s[/cyan])")

            stage_times["⚡ 并行预处理"] = preprocessing_time

            target_lang_output_path = self._save_subtitle_files(
                asr_data, translate_result, input_srt_path, output_dir, target_lang
            )

            total_elapsed = time.time() - task_start_time

            print()
            self._format_time_stats(stage_times, total_elapsed)

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
            self.logger.exception("详细错误信息:")
            raise

    def _extract_context_info(self, input_file: str) -> str:
        """提取上下文信息：外部文件 + 文件名/路径"""
        path = Path(input_file)
        context_parts = []

        external_context = self._read_external_context(path.parent)
        if external_context:
            context_parts.append(external_context)

        folder_path = self._extract_folder_path(path.parent)
        if folder_path:
            context_parts.append(f"Folder path: {folder_path}")

        readable_filename = path.stem.replace('_', ' ').replace('-', ' ')
        context_parts.append(f"Filename: {readable_filename}")

        return "\n\n".join(context_parts)

    def _read_external_context(self, parent_dir: Path) -> str:
        """读取外部上下文文件"""
        for ctx_filename in ['context.txt', 'ctx.txt']:
            ctx_file = parent_dir / ctx_filename
            if ctx_file.exists():
                try:
                    content = ctx_file.read_text(encoding='utf-8').strip()
                    if content:
                        return content
                except Exception:
                    pass
        return ""

    def _extract_folder_path(self, parent_dir: Path, max_depth: int = 3) -> str:
        """提取文件夹路径信息"""
        parent_names = []
        current_path = parent_dir

        for _ in range(max_depth):
            if not current_path.name or current_path.name in ['/', '.', '..']:
                break
            folder_name = current_path.name.replace('_', ' ').replace('-', ' ')
            parent_names.append(folder_name)
            current_path = current_path.parent

        return ' / '.join(reversed(parent_names))

    def _translate_with_pipeline(self, asr_data: SubtitleData, context_info: str) -> Tuple[SubtitleData, list]:
        """
        流水线式翻译：每个批次断句后立即翻译

        Returns:
            (final_asr_data, translate_result)
        """
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

        # 5. 并发处理
        concurrency = self.config.thread_num
        all_translated_results = []
        all_segments = []
        batch_logs_all = []

        def process_batch_task(args):
            """每个批次的完整任务：断句 + 翻译"""
            batch_index, batch = args
            batch_segments = merge_segments_within_batch(
                batch, word_segments,
                model=self.config.split_model,
                batch_index=batch_index + 1
            )

            batch_asr_data = SubtitleData(batch_segments)
            translator = SubtitleOptimizer(config=self.config)
            batch_translate_result = translator.translate_batch_directly(batch_asr_data, context_info)

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
                self.logger.info(f"📈 流水线进度: {progress}/{len(batch_tasks)}")

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

        self.logger.info(f"✅ 流水线处理完成！共 {len(all_segments)} 句")

        return final_asr_data, renumbered_results

    def _print_optimization_details(self, batch_logs: list) -> None:
        """打印详细的优化日志"""
        from .translation_core.optimizer import format_diff

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

        for stage_name, elapsed_time in stages.items():
            if elapsed_time > 0 and stage_name != "⚡ 并行预处理":
                percentage = (elapsed_time / total_time) * 100
                print(f"   {stage_name}: [cyan]{elapsed_time:.1f}s[/cyan] ([dim]{percentage:.0f}%[/dim])")

        print(f"   [bold]总计: [cyan]{total_time:.1f}s[/cyan][/bold]") 