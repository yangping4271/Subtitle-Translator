"""Subtitle segmentation 使用的确定性文本规则。"""

import re


MIN_SENTENCE_WORDS = 3
MIN_LAST_SEGMENT_WORDS = 2


def count_words(text: str) -> int:
    """按英文单词和非拉丁文字字符统计字幕阅读单位。"""
    patterns = [
        r"[\u4e00-\u9fff]",
        r"[\u3040-\u309f]",
        r"[\u30a0-\u30ff]",
        r"[\uac00-\ud7af]",
        r"[\u0e00-\u0e7f]",
        r"[\u0600-\u06ff]",
        r"[\u0400-\u04ff]",
        r"[\u0590-\u05ff]",
        r"[\u1e00-\u1eff]",
        r"[\u3130-\u318f]",
    ]

    non_english_chars = 0
    remaining_text = text
    for pattern in patterns:
        non_english_chars += len(re.findall(pattern, remaining_text))
        remaining_text = re.sub(pattern, " ", remaining_text)

    return non_english_chars + len(remaining_text.strip().split())


def split_by_end_marks(sentence: str) -> list[str]:
    """按句末标记分割，同时保护小数和过短尾段。"""
    positions = []
    for mark in [". ", "! ", "? ", ".", "!", "?"]:
        start = 0
        while True:
            pos = sentence.find(mark, start)
            if pos == -1:
                break
            if _is_decimal_point(sentence, pos, mark):
                start = pos + 1
                continue
            positions.append(pos + len(mark))
            start = pos + 1

    if not positions:
        return [sentence]

    segments = []
    start = 0
    for pos in sorted(set(positions)):
        segment = sentence[start:pos].strip()
        if segment and count_words(segment) >= MIN_SENTENCE_WORDS:
            segments.append(segment)
            start = pos

    last_segment = sentence[start:].strip()
    if last_segment:
        if segments and count_words(last_segment) < MIN_LAST_SEGMENT_WORDS:
            segments[-1] += " " + last_segment
        else:
            segments.append(last_segment)

    return segments if len(segments) > 1 else [sentence]


def _is_decimal_point(sentence: str, pos: int, mark: str) -> bool:
    return mark in (".", ". ") and pos > 0 and sentence[pos - 1].isdigit()
