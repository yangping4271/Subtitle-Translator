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


def test_ass_output_language_suffix_and_styles(tmp_path):
    from subtitle_translator.translation_core.utils.ass_converter import (
        convert_srt_to_ass,
    )

    english = tmp_path / "lesson.en.srt"
    english.write_text(SRT_CONTENT, encoding="utf-8")
    for suffix, expected_name, font in (
        ("zh-cn", "lesson.ass", "宋体-简 黑体,11"),
        ("ja", "lesson.ass", "Noto Sans CJK JP,13"),
        ("unknown", "lesson.unknown.ass", "Noto Sans,13"),
    ):
        target = tmp_path / f"lesson.{suffix}.srt"
        target.write_text(SRT_CONTENT.replace("Hello world", "你好"), encoding="utf-8")
        output = convert_srt_to_ass(target, english, tmp_path)
        assert output.name == expected_name
        content = output.read_text(encoding="utf-8")
        assert f"Style: Secondary,{font}," in content
        assert content.count("Dialogue:") == 4
        assert "你好" in content and "Hello world" in content
