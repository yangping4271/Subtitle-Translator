import pytest

from subtitle_translator.output_files import write_subtitle_outputs
from subtitle_translator.translation_core.data import SubtitleData, SubtitleSegment


def test_write_subtitle_outputs_removes_intermediate_srt_files(tmp_path):
    input_path = tmp_path / "lesson.srt"
    input_path.write_text("source", encoding="utf-8")
    sentence_subtitle = SubtitleData(
        [SubtitleSegment("Hello world!", start_time=0, end_time=1000)]
    )
    translation_results = [
        {
            "id": 1,
            "original": "Hello world!",
            "optimized": "Hello world!",
            "translation": "你好，世界！",
            "discarded": False,
        }
    ]

    outputs = write_subtitle_outputs(
        sentence_subtitle=sentence_subtitle,
        translation_results=translation_results,
        input_srt_path=input_path,
        output_dir=tmp_path / "output",
        target_lang="zh",
        preserve_intermediate=False,
    )

    assert outputs.bilingual_ass.exists()
    assert not outputs.target_srt.exists()
    assert not outputs.source_srt.exists()
    assert outputs.intermediates_preserved is False


@pytest.mark.parametrize("optimized", [None, "", "  ", ".,;:"])
def test_empty_optimized_text_falls_back_without_losing_timestamp(tmp_path, optimized):
    subtitles = SubtitleData([SubtitleSegment("Hello world!", 1234, 5678)])
    source, target = tmp_path / "source.srt", tmp_path / "target.srt"
    subtitles.save_translations_to_files(
        [{"id": 1, "optimized": optimized, "translation": None}],
        str(source),
        str(target),
    )

    assert source.read_text() == "1\n00:00:01,234 --> 00:00:05,678\nHello world!\n"
    assert target.read_text() == "1\n00:00:01,234 --> 00:00:05,678\n\n"
