"""Translation batch 的执行、响应规范化与失败降级。"""

import json
import re
import string
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, TypedDict

from openai import BadRequestError

from ..logger import setup_logger
from .config import SubtitleConfig
from .data import SubtitleData
from .llm_client import ModelAdapter
from .prompts import TRANSLATE_PROMPT
from .translation_retry import _TranslationFallback, _is_translation_failed
from .translation_context import TranslationContext
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
    length_ratio = len(optimized_tokens) / len(original_tokens)

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

    def __init__(
        self,
        config: SubtitleConfig,
        llm: ModelAdapter,
        translation_context: TranslationContext,
    ):
        self.config = config
        self.llm = llm
        self.translation_context = translation_context
        self.executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=config.thread_num
        )
        self.batch_logs = []
        self._fallback = _TranslationFallback(
            config=self.config,
            llm=self.llm,
            translation_context=self.translation_context,
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
        if self.executor is not None:
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
        # 保留默认 JSON 空格：DeepSeek V4 Flash 实测对紧凑格式存在翻译回退。
        source_json = json.dumps(original_subtitle, ensure_ascii=False)
        input_content = f"<subtitles>{source_json}</subtitles>"

        if context_info:
            input_content += f"\n\n<reference>\n{context_info}\n</reference>"

        prompt = TRANSLATE_PROMPT.format(
            target_language=self.translation_context.target_language,
            terminology=self._fallback._format_terminology(
                "\n".join(original_subtitle.values())
            ),
        )

        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_content},
        ]

    def _get_translation_response_format(self) -> dict:
        """按供应商和部署环境选择结构化输出格式。"""
        if self.config.provider_type() == "deepseek":
            return TRANSLATION_JSON_OBJECT_RESPONSE_FORMAT
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
        except BadRequestError as exc:
            # 网络、限流与服务端错误由 SDK 重试；仅格式不受支持时降级。
            error = str(exc).lower()
            if not (
                any(field in error for field in ("response_format", "json_schema", "json_object"))
                and any(marker in error for marker in ("not support", "unsupported", "not available"))
            ):
                raise
            logger.warning(f"⚠️ 结构化输出不受支持，回退到普通模式: {exc}")
            return self.llm.create_chat_completion(**kwargs)

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

        message = self._create_translate_message(
            original_subtitle, context_info
        )
        if self.config.log_raw_payloads:
            logger.debug(
                "输入JSON: %s",
                json.dumps(original_subtitle, ensure_ascii=False),
            )
        response = self._create_chat_completion_with_fallback(message)
        try:
            raw_response = validate_api_response(response, batch_info)
        except ValueError as exc:
            logger.warning("⚠️ %s 响应内容无效: %s", batch_info, exc)
            return self._build_translation_results({}, original_subtitle)
        if self.config.log_raw_payloads:
            logger.debug("%s LLM原始返回数据:\n%s", batch_info, raw_response)
        else:
            logger.info(
                "%s LLM返回摘要: %s 字符（原文日志已关闭）",
                batch_info,
                len(raw_response),
            )

        response_content = parse_translation_response(raw_response)
        missing_ids = original_subtitle.keys() - response_content.keys()
        if missing_ids:
            logger.warning(
                "⚠️ %s LLM丢失ID: %s", batch_info, sorted(map(int, missing_ids))
            )
        return self._build_translation_results(
            response_content, original_subtitle
        )

    def _build_translation_results(
        self, response_content: dict, original_subtitle: dict
    ) -> list:
        """按输入顺序补全结果，保留原文并拒绝跨 ID 的内容搬移。"""
        results = []
        for subtitle_id, original in original_subtitle.items():
            item = response_content.get(subtitle_id, {})
            optimized = item.get("optimized_subtitle")
            shifted = False
            if not isinstance(optimized, str) or not optimized.strip():
                optimized = original
            elif _is_suspicious_optimized_shift(original, optimized):
                logger.warning(
                    "⚠️ 字幕ID %s 的 optimized 疑似跨 ID 错位，将重试翻译",
                    subtitle_id,
                )
                if self.config.log_raw_payloads:
                    logger.debug("原文: %s -> %s", original, optimized)
                optimized = original
                shifted = True

            translation = "" if shifted else item.get("translation", "")
            result = {
                "id": int(subtitle_id),
                "original": original,
                "optimized": optimized,
                "translation": translation if isinstance(translation, str) else "",
                "discarded": not shifted and item.get("discarded") is True,
            }
            results.append(result)
            if original != optimized:
                self.batch_logs.append({"type": "content_optimization", **result})
        return results
