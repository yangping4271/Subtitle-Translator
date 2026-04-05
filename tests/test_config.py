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


def test_config_uses_local_and_remote_default_thread_counts(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.delenv("THREAD_NUM", raising=False)
    local_config = SubtitleConfig()
    assert local_config.thread_num == 4

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    remote_config = SubtitleConfig()
    assert remote_config.thread_num == 18
