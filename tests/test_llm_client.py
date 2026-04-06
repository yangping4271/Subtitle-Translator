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
