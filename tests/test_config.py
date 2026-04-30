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


def test_config_detects_local_openai_compatible_endpoint():
    local_config = SubtitleConfig(
        openai_base_url="http://127.0.0.1:1234/v1",
        _skip_env_load=True,
    )
    remote_config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        _skip_env_load=True,
    )

    assert local_config.is_local_openai_compatible() is True
    assert remote_config.is_local_openai_compatible() is False


def test_disable_thinking_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DISABLE_THINKING", "false")

    config = SubtitleConfig()

    assert config.disable_thinking is False


def test_config_detects_provider_type_from_base_url():
    assert SubtitleConfig(
        openai_base_url="https://api.deepseek.com/v1",
        _skip_env_load=True,
    ).provider_type() == "deepseek"
    assert SubtitleConfig(
        openai_base_url="https://openrouter.ai/api/v1",
        _skip_env_load=True,
    ).provider_type() == "openrouter"
    assert SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        _skip_env_load=True,
    ).provider_type() == "openai"
    assert SubtitleConfig(
        openai_base_url="http://127.0.0.1:1234/v1",
        _skip_env_load=True,
    ).provider_type() == "custom"
