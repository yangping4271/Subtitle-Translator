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


def test_direct_config_construction_does_not_read_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=7,
    )

    assert config.openai_base_url == "https://api.openai.com/v1"
    assert config.thread_num == 7


def test_config_from_env_is_the_explicit_environment_seam(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "local-model")
    monkeypatch.delenv("THREAD_NUM", raising=False)

    config = SubtitleConfig.from_env()

    assert config.openai_base_url == "http://127.0.0.1:1234/v1"
    assert config.split_model == "local-model"
    assert config.translation_model == "local-model"
    assert config.thread_num == 4


def test_config_uses_local_and_remote_default_thread_counts(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.delenv("THREAD_NUM", raising=False)
    local_config = SubtitleConfig.from_env()
    assert local_config.thread_num == 4

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    remote_config = SubtitleConfig.from_env()
    assert remote_config.thread_num == 18


def test_config_detects_local_openai_compatible_endpoint():
    local_config = SubtitleConfig(
        openai_base_url="http://127.0.0.1:1234/v1",
    )
    remote_config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
    )

    assert local_config.is_local_openai_compatible() is True
    assert remote_config.is_local_openai_compatible() is False


def test_disable_thinking_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DISABLE_THINKING", "false")

    config = SubtitleConfig.from_env()

    assert config.disable_thinking is False


def test_raw_payload_logging_is_disabled_by_default_and_can_be_enabled(monkeypatch):
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
    )
    assert config.log_raw_payloads is False

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LOG_RAW_PAYLOADS", "true")
    assert SubtitleConfig.from_env().log_raw_payloads is True


def test_config_detects_provider_type_from_base_url():
    assert SubtitleConfig(
        openai_base_url="https://api.deepseek.com/v1",
    ).provider_type() == "deepseek"
    assert SubtitleConfig(
        openai_base_url="https://openrouter.ai/api/v1",
    ).provider_type() == "openrouter"
    assert SubtitleConfig(
        openai_base_url="https://ai-proxy.chatwise.app/openrouter/api/v1",
    ).provider_type() == "openrouter"
    assert SubtitleConfig(
        openai_base_url="https://ai-proxy.example.com/deepseek/v1",
    ).provider_type() == "deepseek"
    assert SubtitleConfig(
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).provider_type() == "dashscope"
    assert SubtitleConfig(
        openai_base_url=(
            "https://workspace-id.ap-southeast-1.maas.aliyuncs.com/"
            "compatible-mode/v1"
        ),
    ).provider_type() == "dashscope"
    assert SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
    ).provider_type() == "openai"
    assert SubtitleConfig(
        openai_base_url="http://127.0.0.1:1234/v1",
    ).provider_type() == "custom"
