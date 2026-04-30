from subtitle_translator.translation_core.utils.response_parser import (
    parse_translation_response,
)


def test_parse_translation_response_from_structured_json():
    response = """
    {
      "subtitles": [
        {"id": 1, "optimized": "Hello world", "translation": "你好，世界", "discarded": false},
        {"id": 2, "optimized": "Claude Code", "translation": "Claude Code", "discarded": false}
      ]
    }
    """

    result = parse_translation_response(response)

    assert result == {
        "1": {"optimized_subtitle": "Hello world", "translation": "你好，世界", "discarded": False},
        "2": {"optimized_subtitle": "Claude Code", "translation": "Claude Code", "discarded": False},
    }


def test_parse_translation_response_from_translation_only_json():
    response = """
    {
      "subtitles": [
        {"id": 1, "translation": "你好，世界", "discarded": false}
      ]
    }
    """

    result = parse_translation_response(response)

    assert result == {
        "1": {"optimized_subtitle": "", "translation": "你好，世界", "discarded": False},
    }


def test_parse_translation_response_preserves_discarded_marker():
    response = """
    {
      "subtitles": [
        {"id": 48, "optimized": "", "translation": "", "discarded": true}
      ]
    }
    """

    result = parse_translation_response(response)

    assert result == {
        "48": {"optimized_subtitle": "", "translation": "", "discarded": True},
    }


def test_parse_translation_response_falls_back_to_xml():
    response = """
    <subtitle id="1">
    <optimized>Hello world</optimized>
    <translation>你好，世界</translation>
    </subtitle>
    """

    result = parse_translation_response(response)

    assert result == {
        "1": {"optimized_subtitle": "Hello world", "translation": "你好，世界", "discarded": False},
    }
