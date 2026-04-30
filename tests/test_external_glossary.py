from subtitle_translator.translation_core.external_glossary import (
    load_external_terminology,
    select_relevant_external_terms,
)
from subtitle_translator.translation_core.terminology import get_terminology_translation


def test_select_relevant_external_terms_matches_current_text_only():
    external_terms = {
        "Database": {"translation": "数据库", "aliases": []},
        "Framework": {"translation": "框架", "aliases": []},
        "Cache": {"translation": "缓存", "aliases": []},
    }

    selected = select_relevant_external_terms(
        "This lesson uses a database and a framework.",
        external_terms,
        max_terms=5,
    )

    assert list(selected) == ["Framework", "Database"]
    assert get_terminology_translation(selected["Database"]) == "数据库"
    assert "Cache" not in selected


def test_select_relevant_external_terms_respects_max_terms():
    external_terms = {
        "Database": {"translation": "数据库", "aliases": []},
        "Framework": {"translation": "框架", "aliases": []},
    }

    selected = select_relevant_external_terms(
        "Database Framework",
        external_terms,
        max_terms=1,
    )

    assert len(selected) == 1


def test_load_external_terminology_skips_non_simplified_chinese():
    assert load_external_terminology("English", ["programming"]) == {}
