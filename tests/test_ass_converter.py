from subtitle_translator.translation_core.utils.ass_converter import (
    parse_srt_content,
    fix_timestamp_overlaps,
)

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
