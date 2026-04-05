import pytest
from subtitle_translator.translation_core.config import SubtitleConfig, get_target_language


def test_get_target_language_valid():
    assert get_target_language("zh") == "简体中文"
    assert get_target_language("zh-cn") == "简体中文"
    assert get_target_language("zh-tw") == "繁体中文"
    assert get_target_language("ja") == "日文"
    assert get_target_language("ko") == "韩文"
    assert get_target_language("fr") == "法文"
    assert get_target_language("en") == "English"


def test_get_target_language_case_insensitive():
    assert get_target_language("ZH") == "简体中文"
    assert get_target_language("JA") == "日文"


def test_get_target_language_invalid():
    with pytest.raises(ValueError):
        get_target_language("invalid_lang")

    with pytest.raises(ValueError):
        get_target_language("xyz")


def test_config_reads_batch_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("THREAD_NUM", "1")
    monkeypatch.setenv("MIN_BATCH_SENTENCES", "10")
    monkeypatch.setenv("MAX_BATCH_SENTENCES", "12")
    monkeypatch.setenv("TARGET_BATCH_SENTENCES", "11")
    monkeypatch.setenv("MAX_WORD_COUNT_ENGLISH", "17")

    config = SubtitleConfig()

    assert config.thread_num == 1
    assert config.min_batch_sentences == 10
    assert config.max_batch_sentences == 12
    assert config.target_batch_sentences == 11
    assert config.max_word_count_english == 17
