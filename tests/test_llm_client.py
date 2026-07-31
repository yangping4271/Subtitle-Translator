from types import SimpleNamespace
from unittest.mock import Mock

from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core import llm_client
from subtitle_translator.translation_core.llm_client import (
    LLMClient,
    get_response_usage,
)


def test_response_usage_includes_reasoning_tokens_and_request_id():
    response = SimpleNamespace(
        id="chatcmpl-123",
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=40,
            total_tokens=160,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )

    assert get_response_usage(response) == {
        "request_id": "chatcmpl-123",
        "prompt_tokens": 120,
        "completion_tokens": 40,
        "total_tokens": 160,
        "reasoning_tokens": 0,
    }


def test_response_usage_tolerates_missing_third_party_fields():
    assert get_response_usage({"id": "request-1"}) == {
        "request_id": "request-1",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
    }


def test_close_releases_the_owned_openai_client(monkeypatch):
    openai_client = Mock()
    monkeypatch.setattr(llm_client, "OpenAI", Mock(return_value=openai_client))
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)

    client.close()

    openai_client.close.assert_called_once_with()


def test_create_chat_completion_keeps_request_body_unchanged():
    config = SubtitleConfig(
        openai_base_url="http://127.0.0.1:1234/v1",
        openai_api_key="",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="gemma-4-e4b-it",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="gemma-4-e4b-it",
        messages=[{"role": "user", "content": "hello"}],
    )


def test_deepseek_v4_disables_thinking_by_default():
    config = SubtitleConfig(
        openai_base_url="https://api.deepseek.com",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_openrouter_disables_reasoning_by_default():
    config = SubtitleConfig(
        openai_base_url="https://openrouter.ai/api/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="google/gemini-3-flash-preview",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="google/gemini-3-flash-preview",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"reasoning": {"effort": "none"}},
    )


def test_openrouter_proxy_disables_reasoning_by_default():
    config = SubtitleConfig(
        openai_base_url="https://ai-proxy.chatwise.app/openrouter/api/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="qwen/qwen3.6-27b",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="qwen/qwen3.6-27b",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"reasoning": {"effort": "none"}},
    )


def test_dashscope_disables_thinking_by_default():
    config = SubtitleConfig(
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"enable_thinking": False},
    )


def test_dashscope_minimax_disables_thinking_by_default():
    config = SubtitleConfig(
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="minimax-m3",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="minimax-m3",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_openai_gpt_5_1_disables_reasoning():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="gpt-5.1",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="gpt-5.1",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="none",
    )


def test_openai_gpt_5_6_luna_disables_reasoning():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="none",
    )


def test_openrouter_gpt_5_6_luna_disables_reasoning():
    config = SubtitleConfig(
        openai_base_url="https://openrouter.ai/api/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="openai/gpt-5.6-luna",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="openai/gpt-5.6-luna",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"reasoning": {"effort": "none"}},
    )


def test_future_gpt_major_version_disables_reasoning():
    config = SubtitleConfig(
        openai_base_url="https://example.com/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="openai/gpt-6",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="openai/gpt-6",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="none",
    )


def test_original_gpt_5_uses_lowest_supported_reasoning_effort():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="gpt-5",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="gpt-5",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="minimal",
    )


def test_disable_thinking_overrides_explicit_reasoning_options():
    config = SubtitleConfig(
        openai_base_url="https://openrouter.ai/api/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="openai/gpt-5.6-luna",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="high",
        extra_body={"reasoning": {"effort": "high"}},
    )

    create_mock.assert_called_once_with(
        model="openai/gpt-5.6-luna",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"reasoning": {"effort": "none"}},
    )


def test_custom_openai_compatible_gpt_5_6_disables_reasoning():
    config = SubtitleConfig(
        openai_base_url="https://example.com/codex/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="none",
    )


def test_gpt_4_model_does_not_receive_reasoning_effort():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )


def test_disable_thinking_false_keeps_request_body_unchanged():
    config = SubtitleConfig(
        openai_base_url="https://api.deepseek.com",
        openai_api_key="test-key",
        disable_thinking=False,
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hello"}],
    )


def test_disable_thinking_false_keeps_openai_request_body_unchanged():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
        disable_thinking=False,
        _skip_env_load=True,
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "hello"}],
    )
