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
