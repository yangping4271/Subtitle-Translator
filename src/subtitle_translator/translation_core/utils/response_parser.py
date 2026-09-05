"""解析结构化 JSON、旧版 ID 字典和 XML 翻译响应。"""

import json
import re


def parse_translation_response(response: str) -> dict:
    if not response:
        return {}
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(
        r"```(?:json|xml)?\s*([\s\S]*?)```", r"\1", cleaned, flags=re.IGNORECASE
    ).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            subtitle_id: {
                "optimized_subtitle": _extract_tag_content(block, "optimized"),
                "translation": _extract_tag_content(block, "translation"),
                "discarded": _extract_tag_content(block, "discarded").lower() == "true",
            }
            for subtitle_id, block in re.findall(
                r'<subtitle\s+id="([^"]+)"\s*>(.*?)</subtitle>',
                cleaned,
                flags=re.DOTALL | re.IGNORECASE,
            )
        }

    if isinstance(parsed, dict):
        if not isinstance(parsed.get("subtitles"), list):
            return parsed
        parsed = parsed["subtitles"]
    if not isinstance(parsed, list):
        return {}

    results = {}
    for item in parsed:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        optimized = item.get("optimized_subtitle", item.get("optimized", ""))
        translation = item.get("translation", "")
        results[str(item["id"])] = {
            "optimized_subtitle": optimized.strip() if isinstance(optimized, str) else "",
            "translation": translation.strip() if isinstance(translation, str) else "",
            "discarded": item.get("discarded") is True,
        }
    return results


def _extract_tag_content(block: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", block, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""
