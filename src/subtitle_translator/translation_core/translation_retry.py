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
from .terminology import get_terminology_aliases, get_terminology_translation
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
        try:
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
                retry_map.update(self._translate_by_single(still_failed))

            return [retry_map.get(result["id"], result) for result in results]
        except Exception as e:
            logger.warning(f"⚠️ {batch_info}重试失败: {e}")
            return results

    def _translate_by_single(self, subtitle_json: Dict[int, str]) -> dict:
        futures = {
            self.executor.submit(self._translate_single_subtitle, key, value): key
            for key, value in subtitle_json.items()
        }
        results = {}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            key = futures[future]
            try:
                result = future.result()
                if result["translation"].strip():
                    results[key] = {
                        "id": key,
                        "original": subtitle_json[key],
                        **result,
                        "discarded": False,
                    }
            except Exception as e:
                logger.error(f"单条翻译失败，字幕ID: {key}，错误: {e}")
            if completed % 5 == 0 or completed == len(futures):
                logger.info(f"单条翻译进度: {completed}/{len(futures)}")
        return results

    @retry.retry(tries=2)
    def _translate_single_subtitle(self, key: int, value: str) -> Dict:
        """翻译单条字幕（带重试）。"""
        message = [
            {
                "role": "system",
                "content": SINGLE_TRANSLATE_PROMPT.format(
                    target_language=self.translation_context.target_language,
                    terminology=self._format_terminology(value),
                ),
            },
            {"role": "user", "content": value},
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
        external_terms = select_relevant_external_terms(
            source_text,
            self.translation_context.external_terminology,
            self.config.external_glossary_max_terms,
        )
        if not user_terms and not external_terms:
            return ""
        lines = [
            "## Standard Terminology",
            "Use these canonical terms/translations exactly:",
        ]
        correction_lines = []
        for term, entry in user_terms.items():
            translation = get_terminology_translation(entry)
            if translation:
                lines.append(f"- {term} → {translation}")
            for alias in get_terminology_aliases(entry):
                correction_lines.append(f"- {alias} → {term}")

        if external_terms:
            lines.extend([
                "",
                "## Relevant External Terminology",
                "These domain terms were found in the current subtitles. Use them when applicable:",
            ])
            for term, entry in external_terms.items():
                translation = get_terminology_translation(entry)
                if translation:
                    lines.append(f"- {term} → {translation}")

        if correction_lines:
            lines.extend([
                "",
                "## Possible ASR Corrections",
                "When these speech-recognition variants appear, correct them before translation:",
            ])
            lines.extend(correction_lines)
        return "\n".join(lines)
