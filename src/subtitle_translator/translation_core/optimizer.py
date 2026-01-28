import concurrent.futures
import json
import re
import string
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import retry

from ..exceptions import TranslationError
from ..logger import setup_logger
from .config import SubtitleConfig
from .llm_client import LLMClient
from .prompts import SINGLE_TRANSLATE_PROMPT, TRANSLATE_PROMPT
from .utils.api import validate_api_response
from .utils.json_repair import parse_llm_response

logger = setup_logger("subtitle_optimizer")


def _is_format_change_only(original: str, optimized: str) -> bool:
    """判断是否只有格式变化（大小写和标点符号）"""
    original_normalized = original.lower().translate(str.maketrans('', '', string.punctuation))
    optimized_normalized = optimized.lower().translate(str.maketrans('', '', string.punctuation))
    return original_normalized == optimized_normalized


def _is_wrong_replacement(original: str, optimized: str) -> bool:
    """检测是否存在错误的替换（替换了不相关的词）"""
    original_words = set(re.findall(r'\b\w+\b', original.lower()))
    optimized_words = set(re.findall(r'\b\w+\b', optimized.lower()))

    removed_words = original_words - optimized_words
    added_words = optimized_words - original_words

    if removed_words and added_words:
        for removed in removed_words:
            for added in added_words:
                if len(removed) > 3 and len(added) > 3 and not any(c in removed for c in added):
                    return True
    return False


def _is_translation_failed(value) -> bool:
    """检查翻译结果是否为失败状态"""
    if isinstance(value, str):
        return value.startswith("[翻译失败]")
    if isinstance(value, dict):
        return value.get("translation", "").startswith("[翻译失败]")
    return False


def is_sentence_complete(text: str) -> bool:
    """检查句子是否完整"""
    sentence_end_markers = ['.', '!', '?', '。', '！', '？', '…']
    bad_end_words = ["and", "or", "but", "so", "yet", "for", "nor", "in", "on", "at", "to", "with", "by", "as"]

    text = text.strip()
    if not text:
        return True

    if any(text.endswith(marker) for marker in sentence_end_markers):
        return True

    for word in bad_end_words:
        if text.lower().endswith(" " + word) or text.lower() == word:
            return False

    words = text.split()
    if len(words) < 3:
        return False

    return True


def format_diff(original: str, optimized: str) -> str:
    """格式化两个字符串的差异，只显示变化部分"""
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

    result = ''

    if start_diff > 3:
        result += '...'

    result += context_before

    if deleted_part:
        result += f'[-{deleted_part}-]'
    if added_part:
        result += f' [+{added_part}+]'

    result += context_after

    if end_diff_original + 4 < len(original_words):
        result += '...'

    return result.strip()


class SubtitleOptimizer:
    """字幕优化和翻译类"""

    def __init__(self, config: Optional[SubtitleConfig] = None):
        self.config = config or SubtitleConfig()
        self.llm = LLMClient.get_instance(self.config)
        self.client = self.llm.client
        self.thread_num = self.config.thread_num
        self.executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=self.thread_num)
        self.batch_logs = []

    def translate_batch_directly(self, asr_data, summary_content: Dict) -> List[Dict]:
        """直接翻译单个批次（用于流水线模式）"""
        subtitle_json = {str(k): v["original_subtitle"]
                        for k, v in asr_data.to_json().items()}

        results = self._translate(subtitle_json, summary_content, batch_num=1, total_batches=1)

        failed_items = {r['id']: r['original'] for r in results
                        if _is_translation_failed(r.get('translation', ''))}

        if failed_items:
            results = self._retry_failed_translations(failed_items, subtitle_json, summary_content, results)

        return results

    def _retry_failed_translations(self, failed_items: dict, subtitle_json: dict,
                                   summary_content: dict, results: list) -> list:
        """重试失败的翻译（批量重试 → 单条并发）"""
        logger.info(f"发现 {len(failed_items)} 条翻译失败，批量重试")
        try:
            retry_results = self._translate({str(k): v for k, v in failed_items.items()}, summary_content)

            still_failed = {}
            retry_map = {}
            for r in retry_results:
                if not _is_translation_failed(r.get('translation', '')):
                    retry_map[r['id']] = r
                else:
                    still_failed[r['id']] = r['original']

            logger.info(f"批量重试成功 {len(retry_map)}/{len(failed_items)} 条")

            if still_failed:
                logger.info(f"⚡ 批量重试后还有{len(still_failed)}条失败，降级到单条并发翻译")
                single_result = self._translate_by_single(still_failed)

                for k, v in single_result["translated_subtitles"].items():
                    if not v.startswith("[翻译失败]"):
                        retry_map[int(k)] = {
                            "id": int(k),
                            "original": single_result["optimized_subtitles"][k],
                            "optimized": single_result["optimized_subtitles"][k],
                            "translation": v
                        }

            for i, r in enumerate(results):
                if r['id'] in retry_map:
                    results[i] = retry_map[r['id']]

            success_count = len(retry_map)
            logger.info(f"✅ 总共重试成功 {success_count}/{len(failed_items)} 条")

        except Exception as e:
            logger.warning(f"重试失败: {e}")
        return results

    def translate(self, asr_data, summary_content: Dict) -> List[Dict]:
        """翻译字幕"""
        try:
            self.batch_logs.clear()

            subtitle_json = {int(k): v["original_subtitle"]
                            for k, v in asr_data.to_json().items()}

            result = self.translate_multi_thread(subtitle_json, summary_content)

            self._validate_translation_quality(result)

            translated_subtitle = self._format_translation_results(result, subtitle_json)

            self._print_all_batch_logs()
            return translated_subtitle
        finally:
            self.stop()

    def _validate_translation_quality(self, result: dict) -> None:
        """验证翻译结果质量"""
        failed_count = sum(1 for v in result["translated_subtitles"].values()
                          if _is_translation_failed(v))

        if failed_count == len(result["translated_subtitles"]):
            suggestion = "💡 建议：请检查翻译模型名称是否正确，或更换其他可用模型"
            raise TranslationError("所有字幕翻译均失败", suggestion)

        if failed_count > 0:
            total_count = len(result["translated_subtitles"])
            logger.warning(f"⚠️ {failed_count}/{total_count} 条字幕翻译失败")

    def _format_translation_results(self, result: dict, subtitle_json: dict) -> list:
        """格式化翻译结果"""
        translated_subtitle = []
        for k, v in result["optimized_subtitles"].items():
            translated_text = {
                "id": int(k),
                "original": subtitle_json[str(k)],
                "optimized": v,
                "translation": result["translated_subtitles"][k]
            }
            translated_subtitle.append(translated_text)
        return translated_subtitle

    def stop(self):
        """优雅关闭线程池"""
        if hasattr(self, 'executor') and self.executor is not None:
            try:
                logger.info("正在等待线程池任务完成...")
                self.executor.shutdown(wait=True)
                logger.info("线程池已关闭")
            except Exception as e:
                logger.error(f"关闭线程池时发生错误: {e}")
            finally:
                self.executor = None

    def translate_multi_thread(self, subtitle_json: Dict[int, str], summary_content: Optional[Dict] = None):
        """多线程批量翻译字幕（流水线处理，每个批次独立降级）"""
        try:
            result = self._batch_translate(subtitle_json, summary_content=summary_content)
            return result
        except Exception as e:
            logger.error(f"批量翻译完全失败：{e}")
            raise

    def _translate_with_fallback(self, chunk: dict, summary_content: Optional[Dict],
                                 batch_num: int, total_batches: int) -> list:
        """
        单个批次的三级降级翻译

        Level 1: 批量翻译
        Level 2: 批次整体重试（1次）
        Level 3: 单条并发翻译（不重试）
        """
        batch_info = f"[批次{batch_num}/{total_batches}]"

        result = self._translate(chunk, summary_content, batch_num, total_batches)

        failed_items = {item['id']: item['original'] for item in result
                       if _is_translation_failed(item.get('translation', ''))}

        if not failed_items:
            logger.info(f"✅ {batch_info} 翻译完成，无失败")
            return result

        logger.info(f"🔄 {batch_info} 发现 {len(failed_items)} 条失败，批次整体重试")
        retry_result = self._translate(chunk, summary_content, batch_num, total_batches)

        retry_map = {r['id']: r for r in retry_result
                    if not _is_translation_failed(r.get('translation', ''))}
        for i, item in enumerate(result):
            if item['id'] in retry_map:
                result[i] = retry_map[item['id']]

        logger.info(f"📊 {batch_info} 批次重试成功 {len(retry_map)}/{len(failed_items)} 条")

        still_failed = {item['id']: item['original'] for item in result
                       if _is_translation_failed(item.get('translation', ''))}

        if not still_failed:
            return result

        logger.info(f"⚡ {batch_info} 批次重试后还有 {len(still_failed)} 条失败，降级到单条并发")
        single_result = self._translate_by_single_no_retry(still_failed)

        success_count = 0
        for i, item in enumerate(result):
            if str(item['id']) in single_result['translated_subtitles']:
                new_translation = single_result['translated_subtitles'][str(item['id'])]
                if not _is_translation_failed(new_translation):
                    success_count += 1
                result[i]['translation'] = new_translation
                result[i]['optimized'] = single_result['optimized_subtitles'][str(item['id'])]

        logger.info(f"✅ {batch_info} 单条并发成功 {success_count}/{len(still_failed)} 条")

        return result

    def _batch_translate(self, subtitle_json: Dict[int, str], summary_content: Optional[Dict] = None) -> Dict:
        """批量翻译字幕的核心方法（流水线处理，每个批次独立降级）"""
        items = list(subtitle_json.items())[:]

        chunks = []
        i = 0
        adjusted_batch_count = 0

        while i < len(items):
            end_idx = min(i + self.config.max_batch_sentences, len(items))

            if end_idx < len(items):
                _, last_text = items[end_idx - 1]

                if not is_sentence_complete(last_text):
                    adjusted_batch_count += 1

                    complete_idx = end_idx - 1
                    while complete_idx > i and not is_sentence_complete(items[complete_idx - 1][1]):
                        complete_idx -= 1

                    if complete_idx > i:
                        end_idx = complete_idx
                    else:
                        complete_idx = end_idx
                        while complete_idx < len(items) and not is_sentence_complete(items[complete_idx - 1][1]):
                            complete_idx += 1
                            if complete_idx - i > self.config.max_batch_sentences * 1.5:
                                break
                        if complete_idx < len(items):
                            end_idx = complete_idx

            chunk = dict(items[i:end_idx])
            chunks.append(chunk)
            i = end_idx

        logger.info(f"📋 翻译任务规划: {len(chunks)}个批次，每批次约{self.config.max_batch_sentences}条字幕")
        if adjusted_batch_count > 0:
            logger.info(f"🔧 已优化{adjusted_batch_count}个批次边界，确保句子完整性")

        actual_threads = min(len(chunks), self.thread_num)
        logger.info(f"⚡ 并发线程: {actual_threads}个")

        if self.executor is None:
            raise RuntimeError("线程池未初始化")

        futures = []
        chunk_map = {}

        for i, chunk in enumerate(chunks):
            future = self.executor.submit(self._translate_with_fallback, chunk, summary_content, i+1, len(chunks))
            futures.append(future)
            chunk_map[future] = chunk

        optimized_subtitles = {}
        translated_subtitles = {}

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
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

            except Exception as e:
                failed_chunk = chunk_map[future]
                logger.error(f"❌ 批次翻译完全失败（包括所有降级尝试）: {e}")
                for k, v in failed_chunk.items():
                    optimized_subtitles[str(k)] = v
                    translated_subtitles[str(k)] = f"[翻译失败] {v}"

        return {
            "optimized_subtitles": optimized_subtitles,
            "translated_subtitles": translated_subtitles
        }

    def _translate_by_single(self, subtitle_json: Dict[int, str]) -> Dict:
        """使用单条翻译模式处理字幕（并发翻译，带重试）"""
        logger.info(f"开始单条并发翻译 {len(subtitle_json)} 条字幕（并发数: {self.thread_num}）")

        if self.executor is None:
            raise RuntimeError("线程池未初始化")

        futures = {}
        for key, value in subtitle_json.items():
            future = self.executor.submit(self._translate_single_subtitle, key, value)
            futures[future] = key

        optimized_subtitles = {}
        translated_subtitles = {}
        completed = 0
        total = len(futures)

        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            completed += 1
            try:
                result = future.result()
                optimized_subtitles[str(key)] = result["optimized"]
                translated_subtitles[str(key)] = result["translation"]
                if completed % 5 == 0 or completed == total:
                    logger.info(f"单条翻译进度: {completed}/{total}")
            except Exception as e:
                logger.error(f"单条翻译失败，字幕ID: {key}，错误: {e}")
                optimized_subtitles[str(key)] = subtitle_json[key]
                translated_subtitles[str(key)] = f"[翻译失败] {subtitle_json[key]}"

        return {
            "optimized_subtitles": optimized_subtitles,
            "translated_subtitles": translated_subtitles
        }

    def _translate_by_single_no_retry(self, subtitle_json: Dict[int, str]) -> Dict:
        """使用单条翻译模式处理字幕（并发翻译，不重试）"""
        logger.info(f"开始单条并发翻译 {len(subtitle_json)} 条字幕（并发数: {self.thread_num}，不重试）")

        if self.executor is None:
            raise RuntimeError("线程池未初始化")

        futures = {}
        for key, value in subtitle_json.items():
            future = self.executor.submit(self._translate_single_subtitle_no_retry, key, value)
            futures[future] = key

        optimized_subtitles = {}
        translated_subtitles = {}
        completed = 0
        total = len(futures)

        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            completed += 1
            try:
                result = future.result()
                optimized_subtitles[str(key)] = result["optimized"]
                translated_subtitles[str(key)] = result["translation"]
                if completed % 5 == 0 or completed == total:
                    logger.info(f"单条翻译进度: {completed}/{total}")
            except Exception as e:
                logger.error(f"单条翻译失败，字幕ID: {key}，错误: {e}")
                optimized_subtitles[str(key)] = subtitle_json[key]
                translated_subtitles[str(key)] = f"[翻译失败] {subtitle_json[key]}"

        return {
            "optimized_subtitles": optimized_subtitles,
            "translated_subtitles": translated_subtitles
        }

    @retry.retry(tries=2)
    def _translate_single_subtitle(self, key: int, value: str) -> Dict:
        """翻译单条字幕（带重试）"""
        return self._translate_single_subtitle_impl(key, value)

    def _translate_single_subtitle_no_retry(self, key: int, value: str) -> Dict:
        """翻译单条字幕（不重试）"""
        return self._translate_single_subtitle_impl(key, value)

    def _translate_single_subtitle_impl(self, key: int, value: str) -> Dict:
        """翻译单条字幕的实现"""
        try:
            message = [{
                "role": "system",
                "content": SINGLE_TRANSLATE_PROMPT.format(
                    target_language=self.config.target_language,
                    terminology=self._format_terminology()
                )
            }, {
                "role": "user",
                "content": value
            }]

            response = self.client.chat.completions.create(
                model=self.config.translation_model,
                stream=False,
                messages=message,
                temperature=0.7,
                timeout=80
            )

            translate = validate_api_response(response, f"字幕ID {key}").strip()
            logger.info(f"✓ 字幕ID {key} 翻译成功")
            return {
                "optimized": value,
                "translation": translate
            }
        except Exception as e:
            logger.error(f"✗ 字幕ID {key} 翻译失败: {e}")
            return {
                "optimized": value,
                "translation": f"[翻译失败] {value}"
            }

    def _format_terminology(self) -> str:
        """格式化术语表为 prompt 文本"""
        if not self.config.terminology:
            return ""

        lines = ["## Standard Terminology"]
        for term, translation in self.config.terminology.items():
            lines.append(f"- {term} → {translation}")

        return "\n".join(lines)

    def _create_translate_message(self, original_subtitle: Dict[str, str],
                                summary_content: Optional[Dict]):
        """创建翻译提示消息"""
        input_content = (f"Correct and translate the following subtitles into {self.config.target_language}:\n"
                        f"<subtitles>{json.dumps(original_subtitle, ensure_ascii=False)}</subtitles>")

        if summary_content and 'summary' in summary_content:
            try:
                summary_json = parse_llm_response(summary_content.get('summary', '{}'))

                reference_parts = []

                if context := summary_json.get('context'):
                    reference_parts.append(
                        f"Context: {context.get('type', '')} - {context.get('topic', '')}"
                    )

                if corrections := summary_json.get('corrections'):
                    reference_parts.append(
                        f"Apply corrections: {json.dumps(corrections, ensure_ascii=False)}"
                    )

                if reference_parts:
                    input_content += "\n\n<reference>\n" + "\n".join(reference_parts) + "\n</reference>"

            except Exception as e:
                logger.warning(f"Failed to parse summary content: {e}")
                input_content += (f"\n\nReference information:\n"
                                f"<reference>{summary_content.get('summary', '')}</reference>")

        prompt = TRANSLATE_PROMPT.format(
            target_language=self.config.target_language,
            terminology=self._format_terminology()
        )

        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_content}
        ]

    def _print_all_batch_logs(self):
        """统一打印所有批次的日志"""
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
                  summary_content: Optional[Dict], batch_num=None, total_batches=None) -> List[Dict]:
        """翻译字幕"""
        batch_info = f"[批次{batch_num}/{total_batches}]" if batch_num and total_batches else ""
        logger.info(f"🌍 {batch_info} 翻译 {len(original_subtitle)} 条字幕")

        max_retries = 2
        current_try = 0

        while current_try < max_retries:
            try:
                message = self._create_translate_message(original_subtitle, summary_content)

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

                response_content = parse_llm_response(raw_response)

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
        """规范化响应格式（将数组转换为字典）"""
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
        """检查并记录缺失的ID"""
        input_ids = set(original_subtitle.keys())
        output_ids = set(response_content.keys())
        missing_ids = input_ids - output_ids
        if missing_ids:
            logger.warning(f"⚠️ {batch_info} LLM丢失ID: {sorted([int(x) for x in missing_ids])}")

    def _fill_missing_fields(self, response_content: dict, original_subtitle: dict) -> dict:
        """补全缺失的字段"""
        for k in original_subtitle.keys():
            if str(k) not in response_content:
                response_content[str(k)] = {
                    "optimized_subtitle": original_subtitle[str(k)],
                    "translation": f"[翻译失败] {original_subtitle[str(k)]}"
                }
            else:
                if "optimized_subtitle" not in response_content[str(k)]:
                    response_content[str(k)]["optimized_subtitle"] = original_subtitle[str(k)]
                if "translation" not in response_content[str(k)]:
                    response_content[str(k)]["translation"] = f"[翻译失败] {original_subtitle[str(k)]}"
        return response_content

    def _build_translation_results(self, response_content: dict, original_subtitle: dict) -> list:
        """构建翻译结果列表"""
        translated_subtitle = []
        for k, v in response_content.items():
            k = int(k)
            translated_text = {
                "id": k,
                "original": original_subtitle[str(k)],
                "optimized": v["optimized_subtitle"],
                "translation": v["translation"]
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
        """创建失败的翻译结果"""
        return [{
            "id": int(k),
            "original": v,
            "optimized": v,
            "translation": f"[翻译失败] {v}"
        } for k, v in original_subtitle.items()]