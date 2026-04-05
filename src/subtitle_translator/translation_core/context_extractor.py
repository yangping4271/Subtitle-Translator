"""LLM 驱动的上下文提炼。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ..context_loader import (
    build_context_info,
    collect_folder_titles,
    find_nearby_titles,
    to_readable_title,
)
from ..logger import setup_logger
from .config import SubtitleConfig
from .data import SubtitleData
from .llm_client import LLMClient
from .prompts import CONTEXT_EXTRACTION_PROMPT
from .utils.api import validate_api_response

logger = setup_logger("context_extractor")

DEFAULT_MAX_SUBTITLE_EXCERPT_CHARS = 4500
DEFAULT_MAX_SIBLING_TITLES = 12
DEFAULT_MAX_EXCERPT_SEGMENTS = 24

CONTEXT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "subtitle_translation_context",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "domain": {"type": "string"},
                "canonical_names": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "hot_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "corrections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "wrong": {"type": "string"},
                            "correct": {"type": "string"},
                        },
                        "required": ["wrong", "correct"],
                        "additionalProperties": False,
                    },
                },
                "style_notes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "summary",
                "domain",
                "canonical_names",
                "hot_terms",
                "corrections",
                "style_notes",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def extract_context_info(input_file: Path, asr_data: SubtitleData, config: SubtitleConfig) -> str:
    """提炼翻译上下文，失败时回退到本地元数据。"""
    fallback_context = build_context_info(input_file)
    subtitle_text = asr_data.to_txt().strip()
    if not subtitle_text:
        return fallback_context

    payload = build_extraction_payload(input_file, asr_data, fallback_context)
    llm = LLMClient.get_instance(config)

    try:
        response = llm.create_chat_completion(
            model=config.translation_model,
            stream=False,
            messages=[
                {"role": "system", "content": CONTEXT_EXTRACTION_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0.2,
            timeout=80,
            response_format=CONTEXT_RESPONSE_FORMAT,
        )
    except Exception as exc:
        logger.warning(f"⚠️ 结构化上下文提炼失败，回退到普通模式: {exc}")
        response = llm.create_chat_completion(
            model=config.translation_model,
            stream=False,
            messages=[
                {"role": "system", "content": CONTEXT_EXTRACTION_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0.2,
            timeout=80,
        )

    raw_response = validate_api_response(response, "上下文提炼")
    logger.info(f"📥 上下文提炼原始返回:\n{raw_response}")

    extracted = parse_context_response(raw_response)
    if not extracted:
        logger.warning("⚠️ 上下文提炼解析失败，回退到本地元数据")
        return fallback_context

    formatted = format_context_reference(extracted, fallback_context)
    return formatted or fallback_context


def build_extraction_payload(input_file: Path, asr_data: SubtitleData, fallback_context: str) -> str:
    """构建上下文提炼请求。"""
    max_sibling_titles = _read_positive_int_env("CONTEXT_SIBLING_TITLES", DEFAULT_MAX_SIBLING_TITLES)
    all_titles = collect_folder_titles(input_file.parent)
    current_title = to_readable_title(input_file.name)
    sibling_titles = find_nearby_titles(all_titles, current_title, limit=max_sibling_titles)
    subtitle_excerpt, truncated = build_subtitle_excerpt(asr_data)

    payload = {
        "current_file": input_file.name,
        "local_metadata_context": fallback_context,
        "sibling_titles": sibling_titles,
        "subtitle_excerpt_truncated": truncated,
        "subtitle_excerpt": subtitle_excerpt,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_context_response(response: str) -> dict[str, Any]:
    """解析上下文提炼响应。"""
    cleaned = _clean_response(response)
    if not cleaned:
        return {}

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        extracted_json = _extract_json_object(cleaned)
        if not extracted_json:
            return {}
        try:
            parsed = json.loads(extracted_json)
        except json.JSONDecodeError:
            return {}

    if not isinstance(parsed, dict):
        return {}

    return {
        "summary": _coerce_string(parsed.get("summary")),
        "domain": _coerce_string(parsed.get("domain")),
        "canonical_names": _coerce_string_list(parsed.get("canonical_names")),
        "hot_terms": _coerce_string_list(parsed.get("hot_terms")),
        "corrections": _coerce_corrections(parsed.get("corrections")),
        "style_notes": _coerce_string_list(parsed.get("style_notes")),
    }


def format_context_reference(extracted: dict[str, Any], fallback_context: str) -> str:
    """将结构化上下文转为翻译阶段可用的 reference 文本。"""
    lines = ["LLM-extracted translation context."]

    if extracted.get("summary"):
        lines.append(f"Summary: {extracted['summary']}")
    if extracted.get("domain"):
        lines.append(f"Domain: {extracted['domain']}")

    canonical_names = extracted.get("canonical_names", [])
    if canonical_names:
        lines.append("Canonical names:")
        lines.extend(f"- {name}" for name in canonical_names)

    hot_terms = extracted.get("hot_terms", [])
    if hot_terms:
        lines.append("Hot terms:")
        lines.extend(f"- {term}" for term in hot_terms)

    corrections = extracted.get("corrections", [])
    if corrections:
        lines.append("Suggested corrections:")
        lines.extend(f"- {item['wrong']} -> {item['correct']}" for item in corrections)

    style_notes = extracted.get("style_notes", [])
    if style_notes:
        lines.append("Style notes:")
        lines.extend(f"- {note}" for note in style_notes)

    if fallback_context:
        lines.append("Supporting metadata:")
        lines.extend(f"- {line}" for line in fallback_context.splitlines() if line.strip())

    meaningful_lines = [line for line in lines if line.strip()]
    return "\n".join(meaningful_lines)


def _clean_response(text: str) -> str:
    """清理响应文本。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_json_object(text: str) -> str:
    """从文本中截取第一个 JSON 对象。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start:end + 1]


def _coerce_string(value: Any) -> str:
    """将值规范化为字符串。"""
    return value.strip() if isinstance(value, str) else ""


def _coerce_string_list(value: Any) -> list[str]:
    """将值规范化为字符串数组。"""
    if not isinstance(value, list):
        return []

    result = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _coerce_corrections(value: Any) -> list[dict[str, str]]:
    """规范化纠错对。"""
    if not isinstance(value, list):
        return []

    result = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        wrong = _coerce_string(item.get("wrong"))
        correct = _coerce_string(item.get("correct"))
        if not wrong or not correct or wrong.casefold() == correct.casefold():
            continue
        key = (wrong.casefold(), correct.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append({"wrong": wrong, "correct": correct})
    return result


def build_subtitle_excerpt(asr_data: SubtitleData) -> tuple[str, bool]:
    """从整份字幕中抽取覆盖全文件的代表性片段。"""
    if not asr_data.segments:
        return "", False

    max_excerpt_segments = _read_positive_int_env(
        "CONTEXT_EXCERPT_SEGMENTS", DEFAULT_MAX_EXCERPT_SEGMENTS
    )
    max_excerpt_chars = _read_positive_int_env(
        "CONTEXT_EXCERPT_CHARS", DEFAULT_MAX_SUBTITLE_EXCERPT_CHARS
    )
    total_segments = len(asr_data.segments)
    sample_count = min(total_segments, max_excerpt_segments)
    chosen_indices = sorted({
        round(index * (total_segments - 1) / max(sample_count - 1, 1))
        for index in range(sample_count)
    })

    lines = []
    for index in chosen_indices:
        text = asr_data.segments[index].text.strip()
        if text:
            lines.append(f"{index + 1}. {text}")

    excerpt = "\n".join(lines)
    truncated = total_segments > sample_count or len(excerpt) > max_excerpt_chars
    if len(excerpt) > max_excerpt_chars:
        excerpt = excerpt[:max_excerpt_chars].rstrip()

    return excerpt, truncated


def _read_positive_int_env(env_name: str, default: int) -> int:
    """读取正整数环境变量，非法时返回默认值。"""
    raw_value = os.getenv(env_name)
    if not raw_value:
        return default

    try:
        parsed = int(raw_value)
    except ValueError:
        return default

    return max(1, parsed)
