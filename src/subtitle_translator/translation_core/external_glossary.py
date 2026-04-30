"""外部术语库加载与按需命中筛选。"""
import csv
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from .terminology import parse_terminology_entry

logger = logging.getLogger(__name__)

IMMERSIVE_TERMS_RAW_BASE = (
    "https://raw.githubusercontent.com/immersive-translate/terms/main/glossaries"
)
DEFAULT_EXTERNAL_GLOSSARY_DOMAINS = ("programming", "tech", "education")
TARGET_LANGUAGE_TO_IMMERSIVE_LANG = {
    "简体中文": "zh-CN",
}


def _cache_dir() -> Path:
    return Path.home() / ".config" / "subtitle-translator" / "external-glossaries" / "immersive-translate"


def _fetch_glossary_csv(domain: str, lang_code: str, cache_file: Path) -> bool:
    """下载外部术语 CSV 到本地缓存；失败时保持静默降级。"""
    url = f"{IMMERSIVE_TERMS_RAW_BASE}/{domain}_{lang_code}.csv"
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=10) as response:
            cache_file.write_bytes(response.read())
        return True
    except (OSError, urllib.error.URLError) as exc:
        logger.warning(f"外部术语库下载失败: {url} {exc}")
        return False


def _read_csv_terms(csv_path: Path) -> dict:
    terms = {}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                source = (row.get("source") or "").strip()
                target = (row.get("target") or "").strip()
                if source and target:
                    terms[source] = parse_terminology_entry(target)
    except (OSError, csv.Error) as exc:
        logger.warning(f"外部术语库读取失败: {csv_path} {exc}")
    return terms


def load_external_terminology(
    target_language: str,
    domains: Iterable[str] = DEFAULT_EXTERNAL_GLOSSARY_DOMAINS,
) -> dict:
    """加载 Immersive Translate 外部术语库。

    目前只默认支持简体中文，避免把不相关语言的术语引入 prompt。
    """
    lang_code = TARGET_LANGUAGE_TO_IMMERSIVE_LANG.get(target_language)
    if not lang_code:
        return {}

    terms = {}
    for domain in domains:
        normalized_domain = domain.strip()
        if not normalized_domain:
            continue
        cache_file = _cache_dir() / f"{normalized_domain}_{lang_code}.csv"
        if not cache_file.exists() and not _fetch_glossary_csv(normalized_domain, lang_code, cache_file):
            continue
        terms.update(_read_csv_terms(cache_file))
    return terms


def select_relevant_external_terms(
    text: str,
    external_terms: dict,
    max_terms: int,
) -> dict:
    """只选择当前文本中出现的外部术语，避免全量注入上下文。"""
    if not text or not external_terms or max_terms <= 0:
        return {}

    selected = {}
    for term, entry in sorted(external_terms.items(), key=lambda item: (-len(item[0]), item[0].lower())):
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)
        if pattern.search(text):
            selected[term] = entry
            if len(selected) >= max_terms:
                break
    return selected
