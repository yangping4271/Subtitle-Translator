from subtitle_translator.translation_core.utils.ass_converter import (
    parse_srt_content,
    fix_timestamp_overlaps,
)
from subtitle_translator.translation_core.data import normalize_chinese_punctuation

SRT_CONTENT = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
How are you?

"""


def test_parse_srt_content():
    subtitles = parse_srt_content(SRT_CONTENT)
    assert len(subtitles) == 2
    assert subtitles[0]["id"] == "1"
    assert subtitles[0]["start"] == "00:00:01,000"
    assert subtitles[0]["end"] == "00:00:03,000"
    assert subtitles[0]["text"] == "Hello world"
    assert subtitles[1]["id"] == "2"
    assert subtitles[1]["text"] == "How are you?"


def test_fix_timestamp_overlaps():
    subtitles = [
        {"id": "1", "start": "00:00:01,000", "end": "00:00:05,000", "text": "First"},
        {"id": "2", "start": "00:00:03,000", "end": "00:00:06,000", "text": "Second"},
    ]
    fixed, count = fix_timestamp_overlaps(subtitles)
    assert count == 1
    assert fixed[0]["end"] == "00:00:03,000"
    assert fixed[1]["end"] == "00:00:06,000"


def test_fix_timestamp_overlaps_no_overlap():
    subtitles = [
        {"id": "1", "start": "00:00:01,000", "end": "00:00:02,000", "text": "First"},
        {"id": "2", "start": "00:00:03,000", "end": "00:00:04,000", "text": "Second"},
    ]
    fixed, count = fix_timestamp_overlaps(subtitles)
    assert count == 0
    assert fixed[0]["end"] == "00:00:02,000"


def test_normalize_chinese_punctuation_keeps_readability():
    text = "谢谢Andrew。构建记忆系统，赋予智能体持久性；使用Oracle AI数据库。"

    assert normalize_chinese_punctuation(text) == (
        "谢谢 Andrew。构建记忆系统，赋予智能体持久性；使用 Oracle AI 数据库"
    )


def test_normalize_chinese_punctuation_removes_trailing_weak_punctuation():
    assert normalize_chinese_punctuation("但当会话结束时，") == "但当会话结束时"
    assert normalize_chinese_punctuation("用于存储、检索、") == "用于存储、检索"
    assert normalize_chinese_punctuation("刚才在做什么？") == "刚才在做什么？"
