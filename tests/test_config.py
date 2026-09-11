import pytest
from subtitle_translator.translation_core.config import (
    SubtitleConfig,
    get_target_language,
)


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
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=7,
    )

    assert config.openai_base_url == "https://api.openai.com/v1"
    assert config.thread_num == 7


def test_config_from_env_is_the_explicit_environment_seam(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.delenv("THREAD_NUM", raising=False)

    config = SubtitleConfig.from_env()

    assert config.openai_base_url == "https://api.openai.com/v1"
    assert config.llm_model == "test-model"
    assert config.thread_num == 18


def test_config_allows_missing_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert SubtitleConfig.from_env().openai_api_key == ""


def test_config_rejects_missing_models(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="LLM_MODEL"):
        SubtitleConfig.from_env()


def test_config_rejects_blank_model_names(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "   ")
    with pytest.raises(ValueError, match="LLM_MODEL"):
        SubtitleConfig.from_env()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:1234/v1",
        "http://127.0.0.2:1234/v1",
        "http://localhost:1234/v1",
        "http://localhost.:1234/v1",
        "http://[::1]:1234/v1",
        "http://[::ffff:127.0.0.1]:1234/v1",
    ],
)
def test_config_accepts_local_endpoint(monkeypatch, base_url):
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert SubtitleConfig.from_env().openai_base_url == base_url


def test_disable_thinking_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("DISABLE_THINKING", "false")

    config = SubtitleConfig.from_env()

    assert config.disable_thinking is False


def test_raw_payload_logging_is_disabled_by_default_and_can_be_enabled(monkeypatch):
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
    )
    assert config.log_raw_payloads is False

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LOG_RAW_PAYLOADS", "true")
    assert SubtitleConfig.from_env().log_raw_payloads is True


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("https://api.deepseek.com/v1", "deepseek"),
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://ai-proxy.chatwise.app/openrouter/api/v1", "openrouter"),
        ("https://ai-proxy.example.com/deepseek/v1", "deepseek"),
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", "dashscope"),
        (
            "https://workspace-id.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            "dashscope",
        ),
        ("https://api.openai.com/v1", "openai"),
        ("https://open.bigmodel.cn/api/paas/v4/", "zhipu"),
        ("https://api.moonshot.cn/v1", "kimi"),
        ("https://api.minimax.io/v1", "minimax"),
        ("https://api.groq.com/openai/v1", "groq"),
        ("https://generativelanguage.googleapis.com/v1beta/openai/", "google"),
        ("https://api.anthropic.com/v1/", "anthropic"),
        ("https://api.x.ai/v1", "xai"),
        ("https://ark.cn-beijing.volces.com/api/v3", "volcengine"),
        ("https://ark.ap-southeast.bytepluses.com/api/v3", "volcengine"),
        ("https://example.com/v1", "custom"),
    ],
)
def test_config_detects_provider_type_from_base_url(base_url, expected):
    assert SubtitleConfig(openai_base_url=base_url).provider_type() == expected
