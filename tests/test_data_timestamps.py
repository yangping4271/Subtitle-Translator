import pytest
from subtitle_translator.translation_core.data import _parse_srt
from subtitle_translator.exceptions import SubtitleProcessError


def test_single_subtitle_end_before_start():
    srt = "1\n00:00:05,000 --> 00:00:01,000\nHello\n"
    with pytest.raises(SubtitleProcessError, match="结束时间.*早于开始时间"):
        _parse_srt(srt)


def test_two_subtitles_backwards():
    srt = "1\n00:00:05,000 --> 00:00:07,000\nA\n\n2\n00:00:03,000 --> 00:00:06,000\nB\n"
    with pytest.raises(SubtitleProcessError, match="早于第 1 条"):
        _parse_srt(srt)


def test_too_many_duplicate_start_times():
    blocks = "\n\n".join(
        f"{i}\n00:00:00,001 --> 00:00:0{i},000\nLine {i}" for i in range(1, 6)
    )
    with pytest.raises(SubtitleProcessError, match="共享相同开始时间"):
        _parse_srt(blocks)


def test_valid_srt_passes():
    srt = (
        "1\n00:00:01,000 --> 00:00:03,000\nA\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nB\n\n"
        "3\n00:00:07,000 --> 00:00:09,000\nC\n"
    )
    result = _parse_srt(srt)
    assert len(result) == 3
