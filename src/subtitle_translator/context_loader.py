"""
上下文加载模块 - 负责提取和加载翻译上下文信息
"""
from pathlib import Path
import re


LANGUAGE_SUFFIXES = (
    ".zh", ".zh-cn", ".zh-tw",
    ".ja", ".ko", ".th", ".vi",
    ".fr", ".de", ".es", ".pt", ".it", ".ru",
    ".ar", ".en",
)

GENERIC_PATH_PARTS = {
    "users", "user", "home", "downloads", "desktop", "documents",
    "movies", "music", "pictures", "public", "volumes", "mnt",
    "media", "private", "tmp", "var",
}

STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "into", "of",
    "on", "or", "the", "to", "with",
}

SINGLE_WORD_GENERIC_TERMS = {
    "chapter", "course", "developers", "episode", "guide", "introduction",
    "lesson", "part", "series", "tutorial", "video", "welcome", "code",
}

CONTEXT_FILE_SUFFIXES = {
    ".srt", ".ass", ".ssa", ".vtt",
    ".mp4", ".mkv", ".mov", ".avi", ".mp3", ".wav", ".m4a",
    ".txt", ".md",
}


def build_context_info(input_file: Path) -> str:
    """构建完整的上下文信息：外部文件 + 自动提取的文件系统元数据"""
    context_parts = []

    external_context = read_external_context(input_file.parent)
    if external_context:
        context_parts.append(external_context)

    filesystem_context = build_filesystem_context(input_file)
    if filesystem_context:
        context_parts.append(filesystem_context)

    return "\n\n".join(context_parts)


def build_filesystem_context(input_file: Path) -> str:
    """从目录名和同级文件名中提炼对翻译有帮助的上下文。"""
    series_titles = extract_path_context_titles(input_file.parent)
    all_titles = collect_folder_titles(input_file.parent)
    current_title = to_readable_title(input_file.name)
    sequence_label, clean_current_title = split_sequence_label(current_title)
    nearby_titles = find_nearby_titles(all_titles, current_title)
    canonical_names = extract_high_confidence_names(series_titles, all_titles)

    lines = ["Filesystem-derived context from existing files."]

    if series_titles:
        lines.append(f"Series path: {' / '.join(series_titles)}")
    if sequence_label:
        lines.append(f"Sequence label: {sequence_label}")
    lines.append(f"Current title: {clean_current_title}")

    if canonical_names:
        lines.append("High-confidence canonical names:")
        lines.extend(f"- {name}" for name in canonical_names)

    if nearby_titles:
        lines.append("Nearby titles in the same folder:")
        lines.extend(f"- {title}" for title in nearby_titles)

    if canonical_names:
        lines.append(
            "Use the canonical names above to correct likely ASR misspellings "
            "for product names, tools, and other proper nouns."
        )

    return "\n".join(lines)


def read_external_context(parent_dir: Path) -> str:
    """读取外部上下文文件"""
    for ctx_filename in ['context.txt', 'ctx.txt']:
        ctx_file = parent_dir / ctx_filename
        if ctx_file.exists():
            try:
                content = ctx_file.read_text(encoding='utf-8').strip()
                if content:
                    return content
            except Exception:
                pass
    return ""


def extract_folder_path(parent_dir: Path, max_depth: int = 3) -> str:
    """提取有语义价值的文件夹路径信息。"""
    return " / ".join(extract_path_context_titles(parent_dir, max_depth=max_depth))


def extract_path_context_titles(parent_dir: Path, max_depth: int = 3) -> list[str]:
    """提取路径中有语义价值的目录名，忽略系统噪音路径。"""
    parent_names = []
    current_path = parent_dir
    home_parts = {part.casefold() for part in Path.home().parts if part not in {"", "/"}}

    for _ in range(max_depth):
        if not current_path.name or current_path.name in {"/", ".", ".."}:
            break

        raw_name = current_path.name.strip()
        folded = raw_name.casefold()
        parent_folded = current_path.parent.name.casefold()

        is_short_volume_label = parent_folded == "volumes" and re.fullmatch(r"[a-z0-9_-]{1,6}", folded)
        if folded not in GENERIC_PATH_PARTS and folded not in home_parts and not is_short_volume_label:
            folder_name = normalize_spaces(raw_name.replace("_", " ").replace("-", " "))
            if folder_name:
                parent_names.append(folder_name)

        current_path = current_path.parent

    return list(reversed(parent_names))


def collect_folder_titles(parent_dir: Path) -> list[str]:
    """收集同级文件标题，自动去重语言版本和不同封装格式。"""
    try:
        entries = list(parent_dir.iterdir())
    except Exception:
        return []

    titles_by_key = {}
    for entry in entries:
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        if entry.suffix.lower() not in CONTEXT_FILE_SUFFIXES:
            continue

        title = to_readable_title(entry.name)
        if not title:
            continue

        key = normalize_key(title)
        titles_by_key.setdefault(key, title)

    return sorted(titles_by_key.values(), key=natural_sort_key)


def find_nearby_titles(all_titles: list[str], current_title: str, limit: int = 4) -> list[str]:
    """按自然排序选择当前文件附近的标题，提供局部上下文。"""
    if not all_titles:
        return []

    current_key = normalize_key(current_title)
    current_index = None
    for index, title in enumerate(all_titles):
        if normalize_key(title) == current_key:
            current_index = index
            break

    if current_index is None:
        return all_titles[:limit]

    nearby = []
    for index, title in enumerate(all_titles):
        if index == current_index:
            continue
        nearby.append((abs(index - current_index), index, strip_sequence_label(title)[1]))

    nearby.sort(key=lambda item: (item[0], item[1]))
    selected = sorted(nearby[:limit], key=lambda item: item[1])
    return [title for _, _, title in selected]


def extract_high_confidence_names(series_titles: list[str], all_titles: list[str], limit: int = 5) -> list[str]:
    """从目录标题和同级文件名中提炼高置信专名，用于纠正 ASR 错拼。"""
    searchable_titles = [strip_sequence_label(title)[1] for title in all_titles]
    candidates = []

    for series_title in series_titles:
        tokens = re.findall(r"[A-Za-z0-9+#.-]+", series_title)
        max_length = min(4, len(tokens))

        for size in range(max_length, 1, -1):
            for start in range(0, len(tokens) - size + 1):
                phrase_tokens = tokens[start:start + size]
                if any(token.casefold() in STOPWORDS for token in phrase_tokens):
                    continue

                phrase = " ".join(phrase_tokens)
                count = sum(1 for title in searchable_titles if phrase.casefold() in title.casefold())
                if count >= 2:
                    candidates.append((phrase, count))

        if candidates:
            continue

        for token in tokens:
            folded = token.casefold()
            if (
                len(token) < 4
                or folded in STOPWORDS
                or folded in SINGLE_WORD_GENERIC_TERMS
            ):
                continue

            count = sum(1 for title in searchable_titles if re.search(rf"\b{re.escape(token)}\b", title, re.IGNORECASE))
            if count >= 2:
                candidates.append((token, count))

    deduped = []
    seen = set()
    for phrase, count in sorted(candidates, key=lambda item: (-item[1], -len(item[0]), item[0].casefold())):
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((phrase, count))

    return [phrase for phrase, _ in deduped[:limit]]


def to_readable_title(filename: str) -> str:
    """将文件名转换为适合提供给模型的可读标题。"""
    cleaned = filename
    if cleaned.startswith("._"):
        cleaned = cleaned[2:]

    cleaned = re.sub(r"\.[^.]+$", "", cleaned)
    cleaned = strip_language_suffix(cleaned)
    cleaned = normalize_spaces(re.sub(r"[_-]+", " ", cleaned))
    return cleaned


def strip_language_suffix(name: str) -> str:
    """移除文件名中的语言后缀。"""
    stripped = name
    lowered = stripped.casefold()
    for suffix in LANGUAGE_SUFFIXES:
        if lowered.endswith(suffix):
            stripped = stripped[:-len(suffix)]
            lowered = stripped.casefold()
            break
    return stripped


def split_sequence_label(title: str) -> tuple[str | None, str]:
    """分离章节序号和标题正文。"""
    match = re.match(r"^\s*([A-Za-z]{0,3}\d+(?:[._-]\d+)*)\s+(.*)$", title)
    if not match:
        return None, title

    label = normalize_spaces(match.group(1).replace("_", ".").replace("-", "."))
    remaining = normalize_spaces(match.group(2))
    return label, remaining or title


def strip_sequence_label(title: str) -> tuple[str | None, str]:
    """返回去掉章节序号后的标题。"""
    return split_sequence_label(title)


def normalize_spaces(text: str) -> str:
    """统一多余空格。"""
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(text: str) -> str:
    """生成去重和比对使用的规范化 key。"""
    return normalize_spaces(text).casefold()


def natural_sort_key(text: str):
    """自然排序 key，避免 2.10 排在 2.2 前面。"""
    parts = re.split(r"(\d+)", text)
    return [int(part) if part.isdigit() else part.casefold() for part in parts]
