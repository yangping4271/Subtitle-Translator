"""
翻译重试与单条翻译模块 - 负责单条翻译、重试和三级降级策略
"""
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional, Tuple

import retry

from ..logger import setup_logger
from .config import SubtitleConfig
from .llm_client import LLMClient
from .prompts import SINGLE_TRANSLATE_PROMPT
from .utils.api import validate_api_response

logger = setup_logger("translation_executor")


def _is_translation_failed(value) -> bool:
    """检查翻译结果是否为失败状态（空字符串视为失败）。"""
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return not value.get("translation", "").strip()
    return False


class TranslationExecutor:
    """负责单条翻译、重试和三级降级策略。"""

    def __init__(
        self,
        config: SubtitleConfig,
        client,
        executor: ThreadPoolExecutor,
        translate_fn: Callable,
    ):
        self.config = config
        self.client = client
        self.thread_num = config.thread_num
        self.executor = executor
        self._translate = translate_fn

    # ── 三级降级 ──────────────────────────────────────────────────────────────

    def translate_with_fallback(
        self, chunk: dict, context_info: Optional[str], batch_num: int, total_batches: int
    ) -> list:
        """单个批次的三级降级翻译。

        Level 1: 批量翻译
        Level 2: 批次整体重试（1次）
        Level 3: 单条并发翻译（不重试）
        """
        batch_info = f"[批次{batch_num}/{total_batches}]"

        result = self._translate(chunk, context_info, batch_num, total_batches)
        failed_items = self._extract_failed_items(result)

        if not failed_items:
            logger.info(f"✅ {batch_info} 翻译完成，无失败")
            return result

        logger.info(f"🔄 {batch_info} 发现 {len(failed_items)} 条失败，批次整体重试")
        result = self._retry_batch(chunk, context_info, batch_num, total_batches, result)
        still_failed = self._extract_failed_items(result)

        if not still_failed:
            return result

        logger.info(f"⚡ {batch_info} 批次重试后还有 {len(still_failed)} 条失败，降级到单条并发")
        return self._fallback_to_single(still_failed, result, batch_info)

    def _retry_batch(
        self, chunk: dict, context_info: Optional[str],
        batch_num: int, total_batches: int, result: list
    ) -> list:
        """批次整体重试。"""
        retry_result = self._translate(chunk, context_info, batch_num, total_batches)
        retry_map = {r['id']: r for r in retry_result
                     if not _is_translation_failed(r.get('translation', ''))}

        for i, item in enumerate(result):
            if item['id'] in retry_map:
                result[i] = retry_map[item['id']]

        failed_count = len(self._extract_failed_items(result))
        success_count = len(retry_map)
        logger.info(f"📊 批次重试成功 {success_count}/{success_count + failed_count} 条")
        return result

    def _fallback_to_single(self, still_failed: dict, result: list, batch_info: str) -> list:
        """降级到单条并发翻译。"""
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

    def _extract_failed_items(self, result: list) -> dict:
        """提取失败的翻译项。"""
        return {item['id']: item['original'] for item in result
                if _is_translation_failed(item.get('translation', ''))}

    # ── 批次失败重试（translate_batch_directly 使用）────────────────────────

    def retry_failed_translations(self, failed_items: dict, context_info: str, results: list) -> list:
        """重试失败的翻译（批量重试 → 单条并发）。"""
        logger.info(f"发现 {len(failed_items)} 条翻译失败，批量重试")
        try:
            retry_results = self._translate({str(k): v for k, v in failed_items.items()}, context_info)

            retry_map, still_failed = self._categorize_retry_results(retry_results)
            logger.info(f"批量重试成功 {len(retry_map)}/{len(failed_items)} 条")

            if still_failed:
                logger.info(f"⚡ 批量重试后还有{len(still_failed)}条失败，降级到单条并发翻译")
                single_result = self._translate_by_single(still_failed)
                self._merge_single_results(single_result, retry_map)

            self._apply_retry_results(results, retry_map)
            logger.info(f"✅ 总共重试成功 {len(retry_map)}/{len(failed_items)} 条")

        except Exception as e:
            logger.warning(f"重试失败: {e}")
        return results

    def _categorize_retry_results(self, retry_results: list) -> Tuple[dict, dict]:
        """分类重试结果为成功和失败。"""
        retry_map = {}
        still_failed = {}
        for r in retry_results:
            if not _is_translation_failed(r.get('translation', '')):
                retry_map[r['id']] = r
            else:
                still_failed[r['id']] = r['original']
        return retry_map, still_failed

    def _apply_retry_results(self, results: list, retry_map: dict) -> None:
        """应用重试结果到原始结果列表。"""
        for i, r in enumerate(results):
            if r['id'] in retry_map:
                results[i] = retry_map[r['id']]

    def _merge_single_results(self, single_result: dict, retry_map: dict) -> None:
        """合并单条翻译结果到重试映射。"""
        for k, v in single_result["translated_subtitles"].items():
            if v.strip():
                retry_map[int(k)] = {
                    "id": int(k),
                    "original": single_result["optimized_subtitles"][k],
                    "optimized": single_result["optimized_subtitles"][k],
                    "translation": v,
                }

    # ── 单条并发翻译 ──────────────────────────────────────────────────────────

    def _translate_by_single(self, subtitle_json: Dict[int, str]) -> Dict:
        """单条翻译模式（带重试）。"""
        return self._translate_by_single_impl(subtitle_json, with_retry=True)

    def _translate_by_single_no_retry(self, subtitle_json: Dict[int, str]) -> Dict:
        """单条翻译模式（不重试）。"""
        return self._translate_by_single_impl(subtitle_json, with_retry=False)

    def _translate_by_single_impl(self, subtitle_json: Dict[int, str], with_retry: bool) -> Dict:
        """单条翻译的通用实现。"""
        retry_text = "并发翻译" if with_retry else "并发翻译，不重试"
        logger.info(f"开始单条{retry_text} {len(subtitle_json)} 条字幕（并发数: {self.thread_num}）")

        if self.executor is None:
            raise RuntimeError("线程池未初始化")

        translate_func = (
            self._translate_single_subtitle if with_retry
            else self._translate_single_subtitle_no_retry
        )
        futures = {
            self.executor.submit(translate_func, key, value): key
            for key, value in subtitle_json.items()
        }
        return self._collect_single_results(futures, subtitle_json)

    def _collect_single_results(self, futures: dict, subtitle_json: Dict[int, str]) -> Dict:
        """收集单条翻译结果。"""
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
                translated_subtitles[str(key)] = ""

        return {
            "optimized_subtitles": optimized_subtitles,
            "translated_subtitles": translated_subtitles,
        }

    @retry.retry(tries=2)
    def _translate_single_subtitle(self, key: int, value: str) -> Dict:
        """翻译单条字幕（带重试）。"""
        return self._translate_single_subtitle_impl(key, value)

    def _translate_single_subtitle_no_retry(self, key: int, value: str) -> Dict:
        """翻译单条字幕（不重试）。"""
        try:
            return self._translate_single_subtitle_impl(key, value)
        except Exception as e:
            logger.error(f"✗ 字幕ID {key} 翻译失败: {e}")
            return {"optimized": value, "translation": ""}

    def _translate_single_subtitle_impl(self, key: int, value: str) -> Dict:
        """翻译单条字幕的实现。"""
        message = [
            {
                "role": "system",
                "content": SINGLE_TRANSLATE_PROMPT.format(
                    target_language=self.config.target_language,
                    terminology=self._format_terminology(),
                ),
            },
            {"role": "user", "content": value},
        ]

        response = LLMClient.get_instance(self.config).create_chat_completion(
            model=self.config.translation_model,
            stream=False,
            messages=message,
            temperature=0.7,
            timeout=80,
        )

        translate = validate_api_response(response, f"字幕ID {key}").strip()
        logger.info(f"✓ 字幕ID {key} 翻译成功")
        return {"optimized": value, "translation": translate}

    def _format_terminology(self) -> str:
        """格式化术语表为 prompt 文本。"""
        if not self.config.terminology:
            return ""
        lines = ["## Standard Terminology"]
        for term, translation in self.config.terminology.items():
            lines.append(f"- {term} → {translation}")
        return "\n".join(lines)
