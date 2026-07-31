"""Translation batch 的执行、响应规范化与失败降级。"""

import json
import re
import string
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, TypedDict

from ..logger import setup_logger
from .config import SubtitleConfig
from .data import SubtitleData
from .llm_client import ModelAdapter
from .prompts import TRANSLATE_PROMPT
from .translation_retry import _TranslationFallback, _is_translation_failed
from .utils.api import validate_api_response
from .utils.response_parser import parse_translation_response

logger = setup_logger("translation_execution")


class TranslationResult(TypedDict):
    """一个 Sentence segment 的 Translation result。"""

    id: int
    original: str
    optimized: str
    translation: str
    discarded: bool


TRANSLATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "subtitle_translation_batch",
        "schema": {
            "type": "object",
            "properties": {
                "subtitles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "optimized": {"type": "string"},
                            "translation": {"type": "string"},
                            "discarded": {"type": "boolean"},
                        },
                        "required": ["id", "optimized", "translation", "discarded"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["subtitles"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

TRANSLATION_ONLY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "subtitle_translation_batch_translation_only",
        "schema": {
            "type": "object",
            "properties": {
                "subtitles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "translation": {"type": "string"},
                            "discarded": {"type": "boolean"},
                        },
                        "required": ["id", "translation", "discarded"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["subtitles"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

TRANSLATION_JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}


def _is_format_change_only(original: str, optimized: str) -> bool:
    """判断是否只有格式变化（大小写和标点符号）。"""
    remove_punctuation = str.maketrans("", "", string.punctuation)
    original_normalized = original.lower().translate(remove_punctuation)
    optimized_normalized = optimized.lower().translate(remove_punctuation)
    return original_normalized == optimized_normalized


def _is_wrong_replacement(original: str, optimized: str) -> bool:
    """检测是否存在错误的替换（替换了不相关的词）。"""
    original_words = set(re.findall(r"\b\w+\b", original.lower()))
    optimized_words = set(re.findall(r"\b\w+\b", optimized.lower()))

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


def _tokenize_for_similarity(text: str) -> List[str]:
    """提取用于比较 optimized 是否仍对应原字幕的词。"""
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _is_suspicious_optimized_shift(original: str, optimized: str) -> bool:
    """判断 optimized 是否疑似搬移了其他字幕内容。"""
    if not original.strip() or not optimized.strip():
        return False
    if _is_format_change_only(original, optimized):
        return False

    original_tokens = _tokenize_for_similarity(original)
    optimized_tokens = _tokenize_for_similarity(optimized)
    if not original_tokens or not optimized_tokens:
        return False

    original_set = set(original_tokens)
    optimized_set = set(optimized_tokens)
    original_coverage = len(original_set & optimized_set) / len(original_set)
    optimized_extra_ratio = len(optimized_set - original_set) / len(optimized_set)
    length_ratio = len(optimized_tokens) / max(len(original_tokens), 1)

    if original_coverage < 0.45:
        return True
    if length_ratio > 1.8 and optimized_extra_ratio > 0.35:
        return True
    return False


def format_diff(original: str, optimized: str) -> str:
    """格式化两个字符串的差异，只显示变化部分。"""
    if original == optimized:
        return f"无变化: {original}"

    original_words = re.split(r"(\s+)", original)
    optimized_words = re.split(r"(\s+)", optimized)

    start_diff = 0
    while (
        start_diff < len(original_words)
        and start_diff < len(optimized_words)
        and original_words[start_diff] == optimized_words[start_diff]
    ):
        start_diff += 1

    end_diff_original = len(original_words) - 1
    end_diff_optimized = len(optimized_words) - 1
    while (
        end_diff_original >= start_diff
        and end_diff_optimized >= start_diff
        and original_words[end_diff_original] == optimized_words[end_diff_optimized]
    ):
        end_diff_original -= 1
        end_diff_optimized -= 1

    deleted_part = "".join(original_words[start_diff : end_diff_original + 1])
    added_part = "".join(optimized_words[start_diff : end_diff_optimized + 1])

    context_before = "".join(original_words[max(0, start_diff - 3) : start_diff])
    context_after = "".join(
        original_words[
            end_diff_original + 1 : min(len(original_words), end_diff_original + 4)
        ]
    )

    parts = []
    if start_diff > 3:
        parts.append("...")
    parts.append(context_before)
    if deleted_part:
        parts.append(f"[-{deleted_part}-]")
    if added_part:
        parts.append(f" [+{added_part}+]")
    parts.append(context_after)
    if end_diff_original + 4 < len(original_words):
        parts.append("...")

    return "".join(parts).strip()


class TranslationEngine:
    """执行一个 Translation batch 的翻译、响应规范化和失败降级。"""

    def __init__(self, config: SubtitleConfig, llm: ModelAdapter):
        self.config = config
        self.llm = llm
        self.thread_num = self.config.thread_num
        self.executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=self.thread_num
        )
        self.batch_logs = []
        self._fallback = _TranslationFallback(
            config=self.config,
            llm=self.llm,
            executor=self.executor,
            translate_fn=self._translate,
        )

    def translate_batch(
        self,
        translation_batch: SubtitleData,
        context_info: str,
        batch_num: int = 1,
        total_batches: int = 1,
    ) -> List[TranslationResult]:
        """翻译一个 Translation batch。"""
        if self.executor is None:
            raise RuntimeError("TranslationEngine is closed")

        subtitle_json = {
            str(k): v["original_subtitle"]
            for k, v in translation_batch.to_json().items()
        }

        results = self._translate(
            subtitle_json,
            context_info,
            batch_num=batch_num,
            total_batches=total_batches,
        )

        failed_items = {
            r["id"]: r["original"] for r in results if _is_translation_failed(r)
        }

        if failed_items:
            results = self._fallback.retry_failed_translations(
                failed_items,
                context_info,
                results,
                batch_num=batch_num,
                total_batches=total_batches,
            )

        return results

    def close(self) -> None:
        """关闭由当前 TranslationEngine 拥有的线程池。"""
        if hasattr(self, "executor") and self.executor is not None:
            try:
                logger.info("正在等待线程池任务完成...")
                self.executor.shutdown(wait=True)
                logger.info("线程池已关闭")
            except Exception as e:
                logger.error(f"关闭线程池时发生错误: {e}")
            finally:
                self.executor = None

    def __enter__(self) -> "TranslationEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _create_translate_message(
        self, original_subtitle: Dict[str, str], context_info: Optional[str]
    ):
        """创建翻译提示消息。"""
        input_content = (
            f"Correct and translate the following subtitles into {self.config.target_language}.\n"
            "Return a single valid JSON object only, with no markdown or code fences.\n"
            f"<subtitles>{json.dumps(original_subtitle, ensure_ascii=False)}</subtitles>"
        )

        if context_info:
            input_content += f"\n\n<reference>\n{context_info}\n</reference>"

        prompt = TRANSLATE_PROMPT.format(
            target_language=self.config.target_language,
            terminology=self._fallback._format_terminology(
                json.dumps(original_subtitle, ensure_ascii=False)
            ),
            required_fields=self._get_required_response_fields(),
        )

        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_content},
        ]

    def _should_use_translation_only_schema(self) -> bool:
        """本机 OpenAI-compatible 模型只返回翻译文本，减少输出压力。"""
        return self.config.is_local_openai_compatible()

    def _get_required_response_fields(self) -> str:
        """返回当前响应格式要求的字段说明。"""
        if self._should_use_translation_only_schema():
            return "`id`, `translation`, and `discarded`"
        return "`id`, `optimized`, `translation`, and `discarded`"

    def _get_translation_response_format(self) -> dict:
        """按供应商和部署环境选择结构化输出格式。"""
        if self.config.provider_type() == "deepseek":
            return TRANSLATION_JSON_OBJECT_RESPONSE_FORMAT
        if self._should_use_translation_only_schema():
            return TRANSLATION_ONLY_RESPONSE_FORMAT
        return TRANSLATION_RESPONSE_FORMAT

    def _create_chat_completion_with_fallback(self, message):
        """优先使用结构化输出，失败时回退到普通聊天补全。"""
        kwargs = {
            "model": self.config.translation_model,
            "stream": False,
            "messages": message,
            "temperature": 0.7,
            "timeout": 80,
        }

        try:
            return self.llm.create_chat_completion(
                **kwargs,
                response_format=self._get_translation_response_format(),
            )
        except Exception as exc:
            logger.warning(f"⚠️ 结构化翻译输出失败，回退到普通模式: {exc}")
            return self.llm.create_chat_completion(**kwargs)

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

    def _translate(
        self,
        original_subtitle: Dict[str, str],
        context_info: Optional[str],
        batch_num=None,
        total_batches=None,
    ) -> List[Dict]:
        """翻译字幕。"""
        batch_info = (
            f"[批次{batch_num}/{total_batches}]" if batch_num and total_batches else ""
        )
        logger.info(f"🌍 {batch_info} 翻译 {len(original_subtitle)} 条字幕")

        max_retries = 2
        current_try = 0

        while current_try < max_retries:
            try:
                message = self._create_translate_message(
                    original_subtitle, context_info
                )

                logger.info(
                    f"📤 {batch_info} 提交给LLM的字幕数据 (共{len(original_subtitle)}条):"
                )
                input_json = json.dumps(original_subtitle, ensure_ascii=False)
                if self.config.log_raw_payloads:
                    logger.debug(f"   输入JSON: {input_json}")
                else:
                    logger.info(
                        "   输入摘要: %s 字符（原文日志已关闭）",
                        len(input_json),
                    )

                response = self._create_chat_completion_with_fallback(message)
                raw_response = validate_api_response(response, batch_info)
                if self.config.log_raw_payloads:
                    logger.debug(f"{batch_info} LLM原始返回数据:\n{raw_response}")
                else:
                    logger.info(
                        "%s LLM返回摘要: %s 字符（原文日志已关闭）",
                        batch_info,
                        len(raw_response),
                    )

                response_content = parse_translation_response(raw_response)

                response_content = self._normalize_response_format(
                    response_content, batch_info
                )

                if not response_content:
                    current_try += 1
                    if current_try < max_retries:
                        logger.warning(
                            f"⚠️ {batch_info} API返回空结果，重试第{current_try}次"
                        )
                        continue
                    logger.error(f"❌ {batch_info} 重试{max_retries}次仍失败")
                    response_content = {}

                self._check_missing_ids(response_content, original_subtitle, batch_info)

                response_content = self._fill_missing_fields(
                    response_content, original_subtitle
                )

                translated_subtitle = self._build_translation_results(
                    response_content, original_subtitle
                )

                return translated_subtitle

            except Exception as e:
                current_try += 1
                if current_try < max_retries:
                    logger.error(
                        f"❌ {batch_info} 翻译失败，重试第{current_try}次: {e}"
                    )
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
                    item_id = (
                        item.get("id") or item.get("subtitle_id") or item.get("key")
                    )
                    if item_id:
                        new_dict[str(item_id)] = {
                            "optimized_subtitle": item.get(
                                "optimized_subtitle", item.get("optimized", "")
                            ),
                            "translation": item.get("translation", ""),
                        }
            return new_dict if new_dict else {}

        if not isinstance(response_content, dict):
            raise Exception(f"LLM返回格式错误，期望dict，实际{type(response_content)}")

        return response_content

    def _check_missing_ids(
        self, response_content: dict, original_subtitle: dict, batch_info: str
    ) -> None:
        """检查并记录缺失的ID。"""
        input_ids = set(original_subtitle.keys())
        output_ids = set(response_content.keys())
        missing_ids = input_ids - output_ids
        if missing_ids:
            logger.warning(
                f"⚠️ {batch_info} LLM丢失ID: {sorted([int(x) for x in missing_ids])}"
            )

    def _fill_missing_fields(
        self, response_content: dict, original_subtitle: dict
    ) -> dict:
        """补全缺失的字段。"""
        for k in original_subtitle.keys():
            subtitle_id = str(k)
            if subtitle_id not in response_content:
                response_content[str(k)] = {
                    "optimized_subtitle": original_subtitle[str(k)],
                    "translation": "",
                    "discarded": False,
                }
            else:
                current_result = response_content[subtitle_id]

                optimized = current_result.get("optimized_subtitle")
                if not isinstance(optimized, str) or not optimized.strip():
                    current_result["optimized_subtitle"] = original_subtitle[
                        subtitle_id
                    ]
                elif _is_suspicious_optimized_shift(
                    original_subtitle[subtitle_id], optimized
                ):
                    if self.config.log_raw_payloads:
                        logger.warning(
                            "⚠️ 字幕ID %s 的 optimized 疑似跨 ID 错位，"
                            "回退为原文: %s -> %s",
                            subtitle_id,
                            original_subtitle[subtitle_id],
                            optimized,
                        )
                    else:
                        logger.warning(
                            "⚠️ 字幕ID %s 的 optimized 疑似跨 ID 错位，"
                            "已回退为原文（内容日志已关闭）",
                            subtitle_id,
                        )
                    current_result["optimized_subtitle"] = original_subtitle[
                        subtitle_id
                    ]

                translation = current_result.get("translation")
                if translation is None or not isinstance(translation, str):
                    current_result["translation"] = ""

                discarded = current_result.get("discarded")
                if not isinstance(discarded, bool):
                    current_result["discarded"] = False
        return response_content

    def _build_translation_results(
        self, response_content: dict, original_subtitle: dict
    ) -> list:
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
                "translation": v.get("translation", "")
                if isinstance(v.get("translation", ""), str)
                else "",
                "discarded": v.get("discarded", False) is True,
            }
            translated_subtitle.append(translated_text)

            if translated_text["original"] != translated_text["optimized"]:
                self.batch_logs.append(
                    {
                        "type": "content_optimization",
                        "id": k,
                        "original": translated_text["original"],
                        "optimized": translated_text["optimized"],
                    }
                )

        return translated_subtitle

    def _create_failed_results(self, original_subtitle: dict) -> list:
        """创建失败的翻译结果。"""
        return [
            {
                "id": int(k),
                "original": v,
                "optimized": v,
                "translation": "",
                "discarded": False,
            }
            for k, v in original_subtitle.items()
        ]
