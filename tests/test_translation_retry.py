"""测试翻译重试机制"""
from unittest.mock import Mock, patch
import pytest

from subtitle_translator.translation_core.translation_retry import TranslationExecutor
from subtitle_translator.translation_core.config import SubtitleConfig


@pytest.fixture
def mock_config():
    config = SubtitleConfig(
        openai_base_url="http://127.0.0.1:1234/v1",
        _skip_env_load=True,
    )
    config.target_language = "简体中文"
    config.translation_model = "gpt-4o"
    config.thread_num = 2
    config.terminology = {}
    return config


@pytest.fixture
def mock_client():
    return Mock()


@pytest.fixture
def mock_executor():
    return Mock()


@pytest.fixture
def mock_translate_fn():
    return Mock()


@pytest.fixture
def translation_executor(mock_config, mock_client, mock_executor, mock_translate_fn):
    return TranslationExecutor(
        config=mock_config,
        client=mock_client,
        executor=mock_executor,
        translate_fn=mock_translate_fn
    )


def test_retry_triggers_on_exception(translation_executor, mock_client):
    """测试重试装饰器在异常时触发"""
    # 模拟 API 调用失败
    mock_llm = Mock()
    mock_llm.create_chat_completion.side_effect = Exception("API Error")

    # 调用带重试的方法，应该抛出异常（重试 2 次后）
    with patch("subtitle_translator.translation_core.translation_retry.LLMClient.get_instance", return_value=mock_llm):
        with pytest.raises(Exception, match="API Error"):
            translation_executor._translate_single_subtitle(1, "test subtitle")

    # 验证被调用了 2 次（初始 + 1 次重试）
    assert mock_llm.create_chat_completion.call_count == 2


def test_no_retry_returns_empty_on_fail(translation_executor, mock_client):
    """测试不重试版本在失败时返回空字符串"""
    # 模拟 API 调用失败
    mock_llm = Mock()
    mock_llm.create_chat_completion.side_effect = Exception("API Error")

    # 调用不重试的方法，应该返回空翻译
    with patch("subtitle_translator.translation_core.translation_retry.LLMClient.get_instance", return_value=mock_llm):
        result = translation_executor._translate_single_subtitle_no_retry(1, "test subtitle")

    assert result["optimized"] == "test subtitle"
    assert result["translation"] == ""
    # 验证只被调用了 1 次（不重试）
    assert mock_llm.create_chat_completion.call_count == 1
