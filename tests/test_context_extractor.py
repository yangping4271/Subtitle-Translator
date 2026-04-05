from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core.context_extractor import (
    build_extraction_payload,
    extract_context_info,
    format_context_reference,
    parse_context_response,
)
from subtitle_translator.translation_core.data import SubtitleData, SubtitleSegment


def _build_subtitle_data() -> SubtitleData:
    return SubtitleData([
        SubtitleSegment("Welcome to Claude Code.", 0, 1000),
        SubtitleSegment("Today we install Claude Code and use Playwright.", 1000, 2000),
    ])


def test_parse_context_response_handles_json_fence():
    response = """```json
    {
      "summary": "Course about Claude Code for developers",
      "domain": "AI coding workflow",
      "canonical_names": ["Claude Code"],
      "hot_terms": ["Playwright"],
      "corrections": [{"wrong": "Claud Code", "correct": "Claude Code"}],
      "style_notes": ["Keep product names in English."]
    }
    ```"""

    result = parse_context_response(response)

    assert result["summary"] == "Course about Claude Code for developers"
    assert result["canonical_names"] == ["Claude Code"]
    assert result["corrections"] == [{"wrong": "Claud Code", "correct": "Claude Code"}]


def test_format_context_reference_includes_extracted_and_fallback():
    extracted = {
        "summary": "Course about Claude Code",
        "domain": "AI coding course",
        "canonical_names": ["Claude Code"],
        "hot_terms": ["Playwright"],
        "corrections": [{"wrong": "Claud Code", "correct": "Claude Code"}],
        "style_notes": ["Audience is professional developers."],
    }

    result = format_context_reference(extracted, "Filesystem-derived context from existing files.")

    assert "LLM-extracted translation context." in result
    assert "Summary: Course about Claude Code" in result
    assert "Canonical names:" in result
    assert "- Claude Code" in result
    assert "Suggested corrections:" in result
    assert "- Claud Code -> Claude Code" in result
    assert "Supporting metadata:" in result


def test_extract_context_info_uses_llm_result(tmp_path):
    course_dir = tmp_path / "Claude Code for Professional Developers"
    course_dir.mkdir()
    input_file = course_dir / "1.1- Welcome.en.srt"
    input_file.touch()
    (course_dir / "1.2- Installing Claude Code.en.srt").touch()

    config = SubtitleConfig(_skip_env_load=True)
    config.translation_model = "test-model"
    asr_data = _build_subtitle_data()

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="""
        {
          "summary": "Course about Claude Code for professional developers",
          "domain": "AI coding workflow",
          "canonical_names": ["Claude Code"],
          "hot_terms": ["Playwright"],
          "corrections": [{"wrong": "Claud Code", "correct": "Claude Code"}],
          "style_notes": ["Use natural Chinese while preserving product names."]
        }
        """))]
    )

    with patch("subtitle_translator.translation_core.context_extractor.LLMClient.get_instance") as mock_get_instance:
        mock_get_instance.return_value.create_chat_completion.return_value = fake_response
        result = extract_context_info(input_file, asr_data, config)

    assert "Summary: Course about Claude Code for professional developers" in result
    assert "- Claude Code" in result
    assert "- Playwright" in result
    assert "- Claud Code -> Claude Code" in result


def test_extract_context_info_falls_back_when_structured_output_unsupported(tmp_path):
    course_dir = tmp_path / "Claude Code for Professional Developers"
    course_dir.mkdir()
    input_file = course_dir / "1.1- Welcome.en.srt"
    input_file.touch()

    config = SubtitleConfig(_skip_env_load=True)
    config.translation_model = "test-model"
    asr_data = _build_subtitle_data()

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="""
        {
          "summary": "Course about Claude Code",
          "domain": "AI coding workflow",
          "canonical_names": ["Claude Code"],
          "hot_terms": [],
          "corrections": [],
          "style_notes": []
        }
        """))]
    )

    with patch("subtitle_translator.translation_core.context_extractor.LLMClient.get_instance") as mock_get_instance:
        mock_client = mock_get_instance.return_value
        mock_client.create_chat_completion.side_effect = [Exception("unsupported response_format"), fake_response]
        result = extract_context_info(input_file, asr_data, config)

    assert "Summary: Course about Claude Code" in result
    assert "- Claude Code" in result


def test_build_extraction_payload_contains_subtitle_text(tmp_path):
    course_dir = tmp_path / "Course"
    course_dir.mkdir()
    input_file = course_dir / "1.1- Welcome.en.srt"
    input_file.touch()

    payload = build_extraction_payload(
        input_file,
        _build_subtitle_data(),
        "Filesystem-derived context from existing files.",
    )

    assert "Welcome to Claude Code." in payload
    assert "Filesystem-derived context from existing files." in payload
    assert str(input_file.name) in payload
