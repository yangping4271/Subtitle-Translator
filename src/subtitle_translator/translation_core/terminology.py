# -*- coding: utf-8 -*-
"""
术语表加载模块

从外部文件加载术语表，支持全局和局部术语表合并。
"""
from pathlib import Path
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


def parse_terminology_entry(value: str) -> dict:
    """解析术语翻译和可选 ASR alias。

    支持旧格式：
        LLM = 大语言模型 (LLM)

    支持带 alias 的新格式：
        LangChain = LangChain | aliases: land chain, lang chain
    """
    parts = [part.strip() for part in value.split("|")]
    translation = parts[0].strip()
    aliases = []

    for part in parts[1:]:
        key, separator, raw_aliases = part.partition(":")
        if separator and key.strip().lower() in {"alias", "aliases", "asr", "asr aliases"}:
            aliases.extend(
                alias.strip()
                for alias in raw_aliases.replace("，", ",").split(",")
                if alias.strip()
            )

    return {"translation": translation, "aliases": aliases}


def get_terminology_translation(value: Any) -> str:
    """兼容旧字符串格式和新结构化术语格式。"""
    if isinstance(value, dict):
        return str(value.get("translation") or "").strip()
    return str(value or "").strip()


def get_terminology_aliases(value: Any) -> list[str]:
    """返回术语的 ASR alias 列表。"""
    if not isinstance(value, dict):
        return []
    aliases = value.get("aliases") or []
    return [str(alias).strip() for alias in aliases if str(alias).strip()]


def load_terminology_file(file_path: Path) -> dict:
    """读取单个术语表文件

    Args:
        file_path: 术语表文件路径

    Returns:
        {语言: {术语: 翻译}} 格式的字典
    """
    if not file_path.exists():
        return {}

    result = {}
    current_language = None
    content = None

    for encoding in ['utf-8', 'gbk']:
        try:
            content = file_path.read_text(encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            if encoding == 'gbk':
                logger.warning(f"无法读取术语表文件 {file_path}")
                return {}

    if content is None:
        return {}

    for line in content.splitlines():
        line = line.strip()

        if not line or line.startswith('#'):
            continue

        if line.startswith('[') and line.endswith(']'):
            current_language = line[1:-1]
            if current_language not in result:
                result[current_language] = {}
            continue

        if '=' in line and current_language:
            term, translation = line.split('=', 1)
            term = term.strip()
            translation = translation.strip()
            if term and translation:
                result[current_language][term] = parse_terminology_entry(translation)

    return result


def load_terminology(target_language: str, input_file_path: Optional[Path] = None) -> dict:
    """加载术语表：全局 + 局部合并

    Args:
        target_language: 目标语言名称（如 "简体中文"）
        input_file_path: 输入文件路径（用于查找局部术语表）

    Returns:
        {术语: 翻译} 格式的字典
    """
    terminology = {}

    # 1. 加载全局术语表（用户配置目录）
    # 使用 ~/.config/subtitle-translator/terminology.txt
    # 这样在开发模式和安装模式下都能正常工作
    config_dir = Path.home() / '.config' / 'subtitle-translator'
    global_terminology_file = config_dir / 'terminology.txt'

    if global_terminology_file.exists():
        global_terms = load_terminology_file(global_terminology_file)
        terminology.update(global_terms.get(target_language, {}))

    # 2. 加载局部术语表（输入文件同目录）
    if input_file_path:
        local_terminology_file = input_file_path.parent / 'terminology.txt'
        if local_terminology_file.exists():
            local_terms = load_terminology_file(local_terminology_file)
            terminology.update(local_terms.get(target_language, {}))

    return terminology
