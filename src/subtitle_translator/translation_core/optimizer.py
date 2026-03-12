import concurrent.futures
import json
import re
import string
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from ..logger import setup_logger
from .config import SubtitleConfig
from .llm_client import LLMClient
from .prompts import TRANSLATE_PROMPT
from .translation_retry import TranslationExecutor, _is_translation_failed
from .utils.api import validate_api_response
from .utils.response_parser import parse_xml_response

logger = setup_logger("subtitle_optimizer")


def _is_format_change_only(original: str, optimized: str) -> bool:
    """判断是否只有格式变化（大小写和标点符号）。"""
    remove_punctuation = str.maketrans('', '', string.punctuation)
    original_normalized = original.lower().translate(remove_punctuation)
    optimized_normalized = optimized.lower().translate(remove_punctuation)
    return original_normalized == optimized_normalized


def _is_wrong_replacement(original: str, optimized: str) -> bool:
    """检测是否存在错误的替换（替换了不相关的词）。"""
    original_words = set(re.findall(r'\b\w+\b', original.lower()))
    optimized_words = set(re.findall(r'\b\w+\b', optimized.lower()))

    removed_words = original_words - optimized_words
    added_words = optimized_words - original_words

    if not (removed_words and added_words):
        return False

    for removed in removed_words:
        if len(removed) <= 3:
            continue
        for added in added_words:
            if len(added) > 3 and not any(c in removed for c in added):
                return True

    return False


def is_sentence_complete(text: str) -> bool:
    """检查句子是否完整。"""
    sentence_end_markers = ['.', '!', '?', '。', '！', '？', '…']
    bad_end_words = ["and", "or", "but", "so", "yet", "for", "nor", "in", "on", "at", "to", "with", "by", "as"]

    text = text.strip()
    if not text:
        return True

    if any(text.endswith(marker) for marker in sentence_end_markers):
        return True

    text_lower = text.lower()
    if any(text_lower.endswith(" " + word) or text_lower == word for word in bad_end_words):
        return False

    return len(text.split()) >= 3


def format_diff(original: str, optimized: str) -> str:
    """格式化两个字符串的差异，只显示变化部分。"""
    if original == optimized:
        return f"无变化: {original}"

    original_words = re.split(r'(\s+)', original)
    optimized_words = re.split(r'(\s+)', optimized)

    start_diff = 0
    while (start_diff < len(original_words) and
           start_diff < len(optimized_words) and
           original_words[start_diff] == optimized_words[start_diff]):
        start_diff += 1

    end_diff_original = len(original_words) - 1
    end_diff_optimized = len(optimized_words) - 1
    while (end_diff_original >= start_diff and
           end_diff_optimized >= start_diff and
           original_words[end_diff_original] == optimized_words[end_diff_optimized]):
        end_diff_original -= 1
        end_diff_optimized -= 1

    deleted_part = ''.join(original_words[start_diff:end_diff_original + 1])
    added_part = ''.join(optimized_words[start_diff:end_diff_optimized + 1])

    context_before = ''.join(original_words[max(0, start_diff - 3):start_diff])
    context_after = ''.join(original_words[end_diff_original + 1:min(len(original_words), end_diff_original + 4)])

    parts = []
    if start_diff > 3:
        parts.append('...')
    parts.append(context_before)
    if deleted_part:
        parts.append(f'[-{deleted_part}-]')
    if added_part:
        parts.append(f' [+{added_part}+]')
    parts.append(context_after)
    if end_diff_original + 4 < len(original_words):
        parts.append('...')

    return ''.join(parts).strip()


class SubtitleOptimizer:
    """字幕优化和翻译类。"""

    def __init__(self, config: Optional[SubtitleConfig] = None):
        self.config = config or SubtitleConfig()
        self.llm = LLMClient.get_instance(self.config)
        self.client = self.llm.client
        self.thread_num = self.config.thread_num
        self.executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=self.thread_num)
        self.batch_logs = []
        self._executor_obj = TranslationExecutor(
            config=self.config,
            client=self.client,
            executor=self.executor,
            translate_fn=self._translate,
        )

    def translate_batch_directly(self, asr_data, context_info: str) -> List[Dict]:
        """直接翻译单个批次（用于流水线模式）。"""
        subtitle_json = {str(k): v["original_subtitle"]
                        for k, v in asr_data.to_json().items()}

        results = self._translate(subtitle_json, context_info, batch_num=1, total_batches=1)

        failed_items = {r['id']: r['original'] for r in results
                        if _is_translation_failed(r.get('translation', ''))}

        if failed_items:
            results = self._executor_obj.retry_failed_translations(failed_items, context_info, results)

        return results

    def stop(self):
        """优雅关闭线程池。"""
        if hasattr(self, 'executor') and self.executor is not None:
            try:
                logger.info("正在等待线程池任务完成...")
                self.executor.shutdown(wait=True)
                logger.info("线程池已关闭")
            except Exception as e:
                logger.error(f"关闭线程池时发生错误: {e}")
            finally:
                self.executor = None

    def _batch_translate(self, subtitle_json: Dict[int, str], context_info: Optional[str] = None) -> Dict:
        """批量翻译字幕的核心方法（流水线处理，每个批次独立降级）。"""
        items = list(subtitle_json.items())
        chunks = self._create_smart_chunks(items)

        self._log_batch_plan(chunks)

        if self.executor is None:
            raise RuntimeError("线程池未初始化")

        futures = self._submit_batch_tasks(chunks, context_info)
        return self._collect_batch_results(futures)

    def _create_smart_chunks(self, items: list) -> list:
        """创建智能分批（确保句子完整性）。"""
        chunks = []
        i = 0
        adjusted_batch_count = 0

        while i < len(items):
            end_idx = min(i + self.config.max_batch_sentences, len(items))

            if end_idx < len(items):
                _, last_text = items[end_idx - 1]
                if not is_sentence_complete(last_text):
                    adjusted_batch_count += 1
                    end_idx = self._adjust_chunk_boundary(items, i, end_idx)

            chunks.append(dict(items[i:end_idx]))
            i = end_idx

        if adjusted_batch_count > 0:
            logger.info(f"🔧 已优化{adjusted_batch_count}个批次边界，确保句子完整性")

        return chunks

    def _adjust_chunk_boundary(self, items: list, start_idx: int, end_idx: int) -> int:
        """调整分批边界以确保句子完整。"""
        complete_idx = end_idx - 1
        while complete_idx > start_idx and not is_sentence_complete(items[complete_idx - 1][1]):
            complete_idx -= 1

        if complete_idx > start_idx:
            return complete_idx

        complete_idx = end_idx
        max_extension = int(self.config.max_batch_sentences * 1.5)
        while complete_idx < len(items) and not is_sentence_complete(items[complete_idx - 1][1]):
            complete_idx += 1
            if complete_idx - start_idx > max_extension:
                break

        return complete_idx if complete_idx < len(items) else end_idx

    def _log_batch_plan(self, chunks: list) -> None:
        """记录批次规划信息。"""
        logger.info(f"📋 翻译任务规划: {len(chunks)}个批次，每批次约{self.config.max_batch_sentences}条字幕")
        actual_threads = min(len(chunks), self.thread_num)
        logger.info(f"⚡ 并发线程: {actual_threads}个")

    def _submit_batch_tasks(self, chunks: list, context_info: Optional[str]) -> dict:
        """提交批次翻译任务。"""
        futures = []
        chunk_map = {}

        for i, chunk in enumerate(chunks):
            future = self.executor.submit(
                self._executor_obj.translate_with_fallback, chunk, context_info, i + 1, len(chunks)
            )
            futures.append(future)
            chunk_map[future] = chunk

        return {'futures': futures, 'chunk_map': chunk_map}

    def _collect_batch_results(self, futures_data: dict) -> Dict:
        """收集批次翻译结果。"""
        optimized_subtitles = {}
        translated_subtitles = {}

        for future in concurrent.futures.as_completed(futures_data['futures']):
            try:
                result = future.result()
                self._merge_batch_result(result, optimized_subtitles, translated_subtitles)
            except Exception as e:
                self._handle_batch_failure(future, futures_data['chunk_map'],
                                          optimized_subtitles, translated_subtitles, e)

        return {
            "optimized_subtitles": optimized_subtitles,
            "translated_subtitles": translated_subtitles
        }

    def _merge_batch_result(self, result: list, optimized_subtitles: dict,
                           translated_subtitles: dict) -> None:
        """合并单个批次的结果。"""
        for item in result:
            k = str(item["id"])
            optimized_subtitles[k] = item["optimized"]
            if "revised_translation" in item:
                translated_subtitles[k] = {
                    "translation": item["translation"],
                    "revised_translation": item["revised_translation"],
                    "revise_suggestions": item["revise_suggestions"]
                }
            else:
                translated_subtitles[k] = item["translation"]

    def _handle_batch_failure(self, future, chunk_map: dict, optimized_subtitles: dict,
                             translated_subtitles: dict, error: Exception) -> None:
        """处理批次翻译失败。"""
        failed_chunk = chunk_map[future]
        logger.error(f"❌ 批次翻译完全失败（包括所有降级尝试）: {error}")
        for k, v in failed_chunk.items():
            optimized_subtitles[str(k)] = v
            translated_subtitles[str(k)] = ""

    def _create_translate_message(self, original_subtitle: Dict[str, str],
                                context_info: Optional[str]):
        """创建翻译提示消息。"""
        input_content = (f"Correct and translate the following subtitles into {self.config.target_language}:\n"
                        f"<subtitles>{json.dumps(original_subtitle, ensure_ascii=False)}</subtitles>")

        if context_info:
            input_content += f"\n\n<reference>\n{context_info}\n</reference>"

        prompt = TRANSLATE_PROMPT.format(
            target_language=self.config.target_language,
            terminology=self._executor_obj._format_terminology()
        )

        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_content}
        ]

    def _print_all_batch_logs(self):
        """统一打印所有批次的日志。"""
        if not self.batch_logs:
            return

        logger.info("📊 字幕优化结果汇总")

        format_changes = 0
        content_changes = 0
        wrong_changes = 0

        for log in self.batch_logs:
            if log["type"] == "content_optimization":
                id_num = log["id"]
                original = log["original"]
                optimized = log["optimized"]

                if original != optimized:
                    logger.info(f"🔧 字幕ID {id_num} - 内容优化:")
                    logger.info(f"   {format_diff(original, optimized)}")

                    if _is_format_change_only(original, optimized):
                        format_changes += 1
                    elif _is_wrong_replacement(original, optimized):
                        wrong_changes += 1
                    else:
                        content_changes += 1

        logger.info("📈 优化统计:")
        logger.info(f"   格式优化: {format_changes} 项")
        logger.info(f"   内容修改: {content_changes} 项")
        if wrong_changes > 0:
            logger.info(f"   ⚠️ 可疑替换: {wrong_changes} 项")

        total_changes = format_changes + content_changes + wrong_changes
        logger.info(f"   总计修改: {total_changes} 项")
        logger.info("✅ 字幕优化汇总完成")

    def _translate(self, original_subtitle: Dict[str, str],
                  context_info: Optional[str], batch_num=None, total_batches=None) -> List[Dict]:
        """翻译字幕。"""
        batch_info = f"[批次{batch_num}/{total_batches}]" if batch_num and total_batches else ""
        logger.info(f"🌍 {batch_info} 翻译 {len(original_subtitle)} 条字幕")

        max_retries = 2
        current_try = 0

        while current_try < max_retries:
            try:
                message = self._create_translate_message(original_subtitle, context_info)

                logger.info(f"📤 {batch_info} 提交给LLM的字幕数据 (共{len(original_subtitle)}条):")
                logger.info(f"   输入JSON: {json.dumps(original_subtitle, ensure_ascii=False)}")

                response = self.client.chat.completions.create(
                    model=self.config.translation_model,
                    stream=False,
                    messages=message,
                    temperature=0.7,
                    timeout=80
                )
                raw_response = validate_api_response(response, batch_info)
                logger.info(f"{batch_info} LLM原始返回数据:\n{raw_response}")

                response_content = parse_xml_response(raw_response)

                response_content = self._normalize_response_format(response_content, batch_info)

                if not response_content:
                    current_try += 1
                    if current_try < max_retries:
                        logger.warning(f"⚠️ {batch_info} API返回空结果，重试第{current_try}次")
                        continue
                    logger.error(f"❌ {batch_info} 重试{max_retries}次仍失败")
                    response_content = {}

                self._check_missing_ids(response_content, original_subtitle, batch_info)

                response_content = self._fill_missing_fields(response_content, original_subtitle)

                translated_subtitle = self._build_translation_results(response_content, original_subtitle)

                return translated_subtitle

            except Exception as e:
                current_try += 1
                if current_try < max_retries:
                    logger.error(f"❌ {batch_info} 翻译失败，重试第{current_try}次: {e}")
                    continue
                logger.error(f"❌ {batch_info} 重试{max_retries}次仍失败: {e}")
                return self._create_failed_results(original_subtitle)

        return self._create_failed_results(original_subtitle)

    def _normalize_response_format(self, response_content, batch_info: str) -> dict:
        """规范化响应格式（将数组转换为字典）。"""
        if isinstance(response_content, list):
            logger.warning(f"⚠️ {batch_info} LLM返回array，尝试转换")
            new_dict = {}
            for item in response_content:
                if isinstance(item, dict):
                    item_id = item.get('id') or item.get('subtitle_id') or item.get('key')
                    if item_id:
                        new_dict[str(item_id)] = {
                            'optimized_subtitle': item.get('optimized_subtitle', item.get('optimized', '')),
                            'translation': item.get('translation', '')
                        }
            return new_dict if new_dict else {}

        if not isinstance(response_content, dict):
            raise Exception(f"LLM返回格式错误，期望dict，实际{type(response_content)}")

        return response_content

    def _check_missing_ids(self, response_content: dict, original_subtitle: dict, batch_info: str) -> None:
        """检查并记录缺失的ID。"""
        input_ids = set(original_subtitle.keys())
        output_ids = set(response_content.keys())
        missing_ids = input_ids - output_ids
        if missing_ids:
            logger.warning(f"⚠️ {batch_info} LLM丢失ID: {sorted([int(x) for x in missing_ids])}")

    def _fill_missing_fields(self, response_content: dict, original_subtitle: dict) -> dict:
        """补全缺失的字段。"""
        for k in original_subtitle.keys():
            subtitle_id = str(k)
            if subtitle_id not in response_content:
                response_content[str(k)] = {
                    "optimized_subtitle": original_subtitle[str(k)],
                    "translation": ""
                }
            else:
                current_result = response_content[subtitle_id]

                optimized = current_result.get("optimized_subtitle")
                if not isinstance(optimized, str) or not optimized.strip():
                    current_result["optimized_subtitle"] = original_subtitle[subtitle_id]

                translation = current_result.get("translation")
                if translation is None or not isinstance(translation, str):
                    current_result["translation"] = ""
        return response_content

    def _build_translation_results(self, response_content: dict, original_subtitle: dict) -> list:
        """构建翻译结果列表。"""
        translated_subtitle = []
        for key in original_subtitle.keys():
            subtitle_id = str(key)
            v = response_content[subtitle_id]
            k = int(subtitle_id)
            translated_text = {
                "id": k,
                "original": original_subtitle[subtitle_id],
                "optimized": v["optimized_subtitle"],
                "translation": v.get("translation", "") if isinstance(v.get("translation", ""), str) else ""
            }
            translated_subtitle.append(translated_text)

            if translated_text["original"] != translated_text["optimized"]:
                self.batch_logs.append({
                    'type': 'content_optimization',
                    'id': k,
                    'original': translated_text['original'],
                    'optimized': translated_text['optimized']
                })

        return translated_subtitle

    def _create_failed_results(self, original_subtitle: dict) -> list:
        """创建失败的翻译结果。"""
        return [{
            "id": int(k),
            "original": v,
            "optimized": v,
            "translation": ""
        } for k, v in original_subtitle.items()]
