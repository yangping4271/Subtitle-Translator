"""TranslationEngine 内部的批量重试和单条降级 implementation。"""
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

import retry

from ..logger import setup_logger
from .config import SubtitleConfig
from .external_glossary import select_relevant_external_terms
from .llm_client import ModelAdapter
from .prompts import SINGLE_TRANSLATE_PROMPT
from .terminology import get_terminology_aliases, get_terminology_translation, term_matches
from .translation_context import TranslationContext
from .utils.api import validate_api_response

logger = setup_logger("translation_executor")


def _is_translation_failed(value) -> bool:
    """检查翻译结果是否为失败状态（空字符串视为失败）。"""
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        if value.get("discarded") is True:
            return False
        return not value.get("translation", "").strip()
    return False


class _TranslationFallback:
    """TranslationEngine 内部的批量重试与单条降级 implementation。"""

    def __init__(
        self,
        config: SubtitleConfig,
        llm: ModelAdapter,
        translation_context: TranslationContext,
        executor: ThreadPoolExecutor,
        translate_fn: Callable,
    ):
        self.config = config
        self.llm = llm
        self.translation_context = translation_context
        self.executor = executor
        self._translate = translate_fn

    # ── 批次失败重试 ──────────────────────────────────────────────────────────

    def retry_failed_translations(
        self,
        failed_items: dict,
        context_info: str,
        results: list,
        batch_num: Optional[int] = None,
        total_batches: Optional[int] = None,
    ) -> list:
        """重试失败的翻译（批量重试 → 单条并发）。"""
        batch_info = (
            f"[批次{batch_num}/{total_batches}] "
            if batch_num is not None and total_batches is not None
            else ""
        )
        logger.info(f"🔄 {batch_info}发现 {len(failed_items)} 条翻译失败，批量重试")
        retry_results = self._translate(
            {str(k): v for k, v in failed_items.items()},
            context_info,
            batch_num=batch_num,
            total_batches=total_batches,
        )

        retry_map = {r["id"]: r for r in retry_results if not _is_translation_failed(r)}
        still_failed = {
            key: value for key, value in failed_items.items() if key not in retry_map
        }
        logger.info(f"📊 {batch_info}批量重试成功 {len(retry_map)}/{len(failed_items)} 条")

        if still_failed:
            logger.info(f"⚡ {batch_info}降级到单条并发翻译 {len(still_failed)} 条")
            retry_map.update(self._translate_by_single(still_failed, context_info))

        return [retry_map.get(result["id"], result) for result in results]

    def _translate_by_single(self, subtitle_json: Dict[int, str], context_info: str = "") -> dict:
        futures = {
            self.executor.submit(self._translate_single_subtitle, key, value, context_info): key
            for key, value in subtitle_json.items()
        }
        results = {}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            key = futures[future]
            try:
                result = future.result()
                results[key] = {
                    "id": key,
                    "original": subtitle_json[key],
                    **result,
                    "discarded": False,
                }
            except ValueError as e:
                logger.error(f"单条翻译失败，字幕ID: {key}，错误: {e}")
            if completed % 5 == 0 or completed == len(futures):
                logger.info(f"单条翻译进度: {completed}/{len(futures)}")
        return results

    @retry.retry(exceptions=ValueError, tries=2)
    def _translate_single_subtitle(self, key: int, value: str, context_info: str = "") -> Dict:
        """翻译单条字幕（带重试）。"""
        message = [
            {
                "role": "system",
                "content": SINGLE_TRANSLATE_PROMPT.format(
                    target_language=self.translation_context.target_language,
                    terminology=self._format_terminology(value),
                ),
            },
            {"role": "user", "content": f"<subtitles>{value}</subtitles>"
             + (f"\n<reference>{context_info}</reference>" if context_info else "")},
        ]

        response = self.llm.create_chat_completion(
            model=self.config.translation_model,
            stream=False,
            messages=message,
            temperature=0.7,
            timeout=80,
        )

        translate = validate_api_response(response, f"字幕ID {key}").strip()
        logger.info(f"✓ 字幕ID {key} 翻译成功")
        return {"optimized": value, "translation": translate}

    def _format_terminology(self, source_text: str = "") -> str:
        """格式化术语表为 prompt 文本。"""
        user_terms = self.translation_context.terminology
        user_lines = []
        corrections = []
        for term, entry in user_terms.items():
            aliases = list(dict.fromkeys(
                alias for alias in get_terminology_aliases(entry)
                if term_matches(source_text, alias)
            ))
            if not aliases and not term_matches(source_text, term):
                continue
            translation = get_terminology_translation(entry)
            if translation and translation != term:
                user_lines.append(f"{term} → {translation}")
            elif not aliases:
                user_lines.append(f"{term} (keep)")
            corrections.extend(f"{alias} → {term}" for alias in aliases)

        user_names = {term.casefold() for term in user_terms}
        external_terms = select_relevant_external_terms(
            source_text,
            {term: entry for term, entry in self.translation_context.external_terminology.items()
             if term.casefold() not in user_names},
            self.config.external_glossary_max_terms,
        )
        external_lines = [
            f"{term} → {get_terminology_translation(entry)}"
            for term, entry in external_terms.items()
            if get_terminology_translation(entry)
            and get_terminology_translation(entry).casefold() != term.casefold()
        ]
        sections = []
        for title, lines in (("User terms", user_lines),
                             ("ASR aliases", corrections),
                             ("External suggestions (context-dependent)", external_lines)):
            if lines:
                sections.append(title + ":\n" + "\n".join(lines))
        return "\n".join(sections)
