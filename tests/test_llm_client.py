from unittest.mock import Mock

from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core.llm_client import LLMClient


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


def test_openai_gpt_5_1_uses_reasoning_effort_none():
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
