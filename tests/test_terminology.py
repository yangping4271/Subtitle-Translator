from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core.terminology import (
    get_terminology_aliases,
    get_terminology_translation,
    load_terminology_file,
    parse_terminology_entry,
)
from subtitle_translator.translation_core.translation_retry import TranslationExecutor


def test_parse_terminology_entry_supports_aliases():
    entry = parse_terminology_entry(
        "LangChain | aliases: land chain, lang chain，length chain"
    )

    assert get_terminology_translation(entry) == "LangChain"
    assert get_terminology_aliases(entry) == [
        "land chain",
        "lang chain",
        "length chain",
    ]


def test_load_terminology_file_keeps_old_format_and_alias_format(tmp_path):
    terminology_file = tmp_path / "terminology.txt"
    terminology_file.write_text(
        "\n".join([
            "[简体中文]",
            "LLM = 大语言模型 (LLM)",
            "LangChain = LangChain | aliases: land chain, lang chain",
        ]),
        encoding="utf-8",
    )

    terms = load_terminology_file(terminology_file)["简体中文"]

    assert get_terminology_translation(terms["LLM"]) == "大语言模型 (LLM)"
    assert get_terminology_aliases(terms["LLM"]) == []
    assert get_terminology_translation(terms["LangChain"]) == "LangChain"
    assert get_terminology_aliases(terms["LangChain"]) == ["land chain", "lang chain"]


def test_format_terminology_includes_asr_corrections():
    config = SubtitleConfig(openai_base_url="https://api.openai.com/v1", _skip_env_load=True)
    config.terminology = {
        "LangChain": {
            "translation": "LangChain",
            "aliases": ["land chain", "lang chain"],
        },
        "LLM": "大语言模型 (LLM)",
    }
    executor = object.__new__(TranslationExecutor)
    executor.config = config

    formatted = executor._format_terminology()

    assert "LangChain → LangChain" in formatted
    assert "LLM → 大语言模型 (LLM)" in formatted
    assert "Possible ASR Corrections" in formatted
    assert "land chain → LangChain" in formatted
    assert "lang chain → LangChain" in formatted


def test_format_terminology_includes_only_relevant_external_terms():
    config = SubtitleConfig(openai_base_url="https://api.openai.com/v1", _skip_env_load=True)
    config.terminology = {}
    config.external_terminology = {
        "Database": {"translation": "数据库", "aliases": []},
        "Framework": {"translation": "框架", "aliases": []},
        "Cache": {"translation": "缓存", "aliases": []},
    }
    config.external_glossary_max_terms = 10
    executor = object.__new__(TranslationExecutor)
    executor.config = config

    formatted = executor._format_terminology("We use a database in this course.")

    assert "Relevant External Terminology" in formatted
    assert "Database → 数据库" in formatted
    assert "Framework → 框架" not in formatted
    assert "Cache → 缓存" not in formatted
