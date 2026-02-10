"""LLM 响应解析工具。"""

import json
import re
from typing import Dict

from ...logger import setup_logger

logger = setup_logger("response_parser")


def parse_xml_response(response: str) -> Dict[str, Dict[str, str]]:
    """解析 LLM 返回的 XML 响应。

    Returns:
        解析后的字典，格式为 {subtitle_id: {optimized_subtitle, translation}}
        解析失败返回空字典
    """
    if not response:
        return {}

    cleaned = _clean_response(response)

    if _is_json_format(cleaned):
        return _parse_json_fallback(cleaned)

    return _parse_xml_format(cleaned)


def _clean_response(text: str) -> str:
    """清理响应文本，移除 think 标签和代码围栏。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:xml)?\s*([\s\S]*?)```", r"\1", text, flags=re.IGNORECASE)
    return text.strip()


def _is_json_format(text: str) -> bool:
    """检查文本是否为 JSON 格式。"""
    return text.lstrip().startswith("{")


def _parse_json_fallback(text: str) -> Dict[str, Dict[str, str]]:
    """尝试解析 JSON 格式的响应（降级处理）。"""
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        logger.warning(f"JSON 回退解析失败: {exc}")
        return {}


def _parse_xml_format(text: str) -> Dict[str, Dict[str, str]]:
    """解析 XML 格式的响应。"""
    try:
        matches = re.findall(
            r'<subtitle\s+id="([^"]+)"\s*>(.*?)</subtitle>',
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    except re.error as exc:
        logger.error(f"XML 正则解析失败: {exc}")
        return {}

    if not matches:
        return {}

    results = {}
    for subtitle_id, block in matches:
        results[str(subtitle_id)] = {
            "optimized_subtitle": _extract_tag_content(block, "optimized"),
            "translation": _extract_tag_content(block, "translation"),
        }

    return results


def _extract_tag_content(block: str, tag: str) -> str:
    """从 XML 块中提取指定标签的内容。"""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", block, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""
