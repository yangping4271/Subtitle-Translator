from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core import llm_client
from subtitle_translator.translation_core.llm_client import (
    LLMClient,
    RequestMetric,
    get_response_usage,
    summarize_request_metrics,
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


def _completion_response(reasoning_tokens: int):
    return SimpleNamespace(
        id="chatcmpl-reasoning",
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=40,
            total_tokens=160,
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens
            ),
        ),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="translated"),
            )
        ],
    )


def test_detected_reasoning_warns_once_per_model(monkeypatch):
    config = SubtitleConfig(
        openai_base_url="https://api.deepseek.com",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    client._client.chat.completions.create = Mock(
        return_value=_completion_response(reasoning_tokens=5849)
    )
    terminal_notice = Mock()
    log_warning = Mock()
    monkeypatch.setattr(llm_client, "rich_print", terminal_notice)
    monkeypatch.setattr(llm_client.logger, "warning", log_warning)

    for _ in range(2):
        client.create_chat_completion(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "hello"}],
        )

    terminal_notice.assert_called_once()
    notice = terminal_notice.call_args.args[0]
    assert "检测到模型实际使用了思考模式（关闭参数未生效）" in notice
    assert "deepseek-v4-flash" in notice
    assert "reasoning_tokens=5849" in notice
    log_warning.assert_called_once_with(
        "检测到模型实际使用了思考模式（关闭参数未生效）: "
        "deepseek-v4-flash, reasoning_tokens=5849"
    )


def test_zero_reasoning_tokens_does_not_show_notice(monkeypatch):
    config = SubtitleConfig(
        openai_base_url="https://api.deepseek.com",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    client._client.chat.completions.create = Mock(
        return_value=_completion_response(reasoning_tokens=0)
    )
    terminal_notice = Mock()
    monkeypatch.setattr(llm_client, "rich_print", terminal_notice)

    client.create_chat_completion(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hello"}],
    )

    terminal_notice.assert_not_called()


def test_request_metrics_merge_concurrent_time_for_effective_throughput():
    metrics = [
        RequestMetric(
            started_at=0.0,
            ended_at=10.0,
            latency=10.0,
            success=True,
            completion_tokens=100,
            finish_reason="stop",
            content_chars=300,
            error_type=None,
        ),
        RequestMetric(
            started_at=0.0,
            ended_at=20.0,
            latency=20.0,
            success=True,
            completion_tokens=200,
            finish_reason="stop",
            content_chars=600,
            error_type=None,
        ),
    ]

    stats = summarize_request_metrics(metrics)

    assert stats["requests"] == 2
    assert stats["successful_requests"] == 2
    assert stats["failed_requests"] == 0
    assert stats["latency_avg"] == 15.0
    assert stats["latency_p95"] == 19.5
    assert stats["latency_max"] == 20.0
    assert stats["effective_tps"] == 15.0
    assert stats["slow_requests"] == 0
    assert stats["anomalies"] == 0
    assert stats["max_context_tokens"] is None
    assert stats["context_reference_tokens"] == 4096
    assert stats["max_context_ratio"] is None


def test_request_metrics_report_longest_context_against_4k():
    metrics = [
        RequestMetric(
            started_at=0.0,
            ended_at=5.0,
            latency=5.0,
            success=True,
            completion_tokens=274,
            finish_reason="stop",
            content_chars=1158,
            error_type=None,
            prompt_tokens=865,
            total_tokens=1139,
        ),
        RequestMetric(
            started_at=5.0,
            ended_at=22.0,
            latency=17.0,
            success=True,
            completion_tokens=1180,
            finish_reason="stop",
            content_chars=3685,
            error_type=None,
            prompt_tokens=1411,
            total_tokens=2591,
        ),
    ]

    stats = summarize_request_metrics(metrics)

    assert stats["max_context_tokens"] == 2591
    assert stats["context_reference_tokens"] == 4096
    assert stats["max_context_ratio"] == 2591 / 4096


def test_request_metrics_report_slow_failures_and_response_anomalies():
    metrics = [
        RequestMetric(
            started_at=0.0,
            ended_at=31.0,
            latency=31.0,
            success=True,
            completion_tokens=None,
            finish_reason="length",
            content_chars=0,
            error_type=None,
        ),
        RequestMetric(
            started_at=32.0,
            ended_at=65.0,
            latency=33.0,
            success=False,
            completion_tokens=None,
            finish_reason=None,
            content_chars=0,
            error_type="APITimeoutError",
        ),
    ]

    stats = summarize_request_metrics(metrics)

    assert stats["requests"] == 2
    assert stats["successful_requests"] == 1
    assert stats["failed_requests"] == 1
    assert stats["effective_tps"] is None
    assert stats["latency_avg"] == 32.0
    assert stats["latency_p95"] == 32.9
    assert stats["latency_max"] == 33.0
    assert stats["slow_requests"] == 2
    assert stats["slowest_request"] == 33.0
    assert stats["empty_responses"] == 1
    assert stats["abnormal_finishes"] == 1
    assert stats["missing_usage"] == 1
    assert stats["anomalies"] == 1
    assert stats["error_types"] == {"APITimeoutError": 1}


def test_close_releases_the_owned_openai_client(monkeypatch):
    openai_client = Mock()
    monkeypatch.setattr(llm_client, "OpenAI", Mock(return_value=openai_client))
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)

    client.close()

    openai_client.close.assert_called_once_with()


def test_create_chat_completion_keeps_request_body_unchanged():
    config = SubtitleConfig(
        openai_base_url="https://example.com/v1",
        openai_api_key="test-key",
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


def test_openrouter_disables_reasoning_for_all_models():
    config = SubtitleConfig(
        openai_base_url="https://openrouter.ai/api/v1",
        openai_api_key="test-key",
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


def test_openrouter_proxy_disables_reasoning_for_all_models():
    config = SubtitleConfig(
        openai_base_url="https://ai-proxy.chatwise.app/openrouter/api/v1",
        openai_api_key="test-key",
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


def test_official_zhipu_disables_thinking_for_all_models():
    config = SubtitleConfig(
        openai_base_url="https://open.bigmodel.cn/api/paas/v4/",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="glm-4-flash",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="glm-4-flash",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_official_minimax_disables_thinking_for_all_models():
    config = SubtitleConfig(
        openai_base_url="https://api.minimax.io/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="MiniMax-M3",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="MiniMax-M3",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_official_kimi_only_disables_registered_models():
    config = SubtitleConfig(
        openai_base_url="https://api.moonshot.cn/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="kimi-k2.6",
        messages=[{"role": "user", "content": "hello"}],
    )
    create_mock.assert_called_once_with(
        model="kimi-k2.6",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"thinking": {"type": "disabled"}},
    )

    create_mock.reset_mock()
    client.create_chat_completion(
        model="kimi-k3",
        messages=[{"role": "user", "content": "hello"}],
    )
    create_mock.assert_called_once_with(
        model="kimi-k3",
        messages=[{"role": "user", "content": "hello"}],
    )


def test_official_google_disables_registered_gemini():
    config = SubtitleConfig(
        openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="gemini-3.6-flash",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="gemini-3.6-flash",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={
            "extra_body": {
                "google": {
                    "thinking_config": {
                        "thinking_level": "minimal",
                    }
                }
            }
        },
    )

    create_mock.reset_mock()
    client.create_chat_completion(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hello"}],
    )
    create_mock.assert_called_once_with(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={
            "extra_body": {
                "google": {
                    "thinking_config": {
                        "thinking_budget": 0,
                    }
                }
            }
        },
    )


def test_official_groq_does_not_disable_thinking():
    config = SubtitleConfig(
        openai_base_url="https://api.groq.com/openai/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "hello"}],
    )
    create_mock.assert_called_once_with(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.reset_mock()
    client.create_chat_completion(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "hello"}],
    )
    create_mock.assert_called_once_with(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "hello"}],
    )


def test_custom_endpoint_glm_uses_registry():
    config = SubtitleConfig(
        openai_base_url="https://example.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="glm-5.2",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="glm-5.2",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_official_dashscope_disables_thinking_for_all_models():
    config = SubtitleConfig(
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="qwen-vl-plus",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="qwen-vl-plus",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"enable_thinking": False},
    )


def test_official_volcengine_disables_thinking_for_all_models():
    config = SubtitleConfig(
        openai_base_url="https://ark.cn-beijing.volces.com/api/v3",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="ep-unregistered",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="ep-unregistered",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_official_anthropic_only_disables_registered_models():
    config = SubtitleConfig(
        openai_base_url="https://api.anthropic.com/v1/",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hello"}],
    )
    create_mock.assert_called_once_with(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"thinking": {"type": "disabled"}},
    )

    create_mock.reset_mock()
    client.create_chat_completion(
        model="claude-fable-5",
        messages=[{"role": "user", "content": "hello"}],
    )
    create_mock.assert_called_once_with(
        model="claude-fable-5",
        messages=[{"role": "user", "content": "hello"}],
    )


def test_official_xai_only_disables_registered_models():
    config = SubtitleConfig(
        openai_base_url="https://api.x.ai/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="grok-4.3",
        messages=[{"role": "user", "content": "hello"}],
    )
    create_mock.assert_called_once_with(
        model="grok-4.3",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="none",
    )

    create_mock.reset_mock()
    client.create_chat_completion(
        model="grok-4.6",
        messages=[{"role": "user", "content": "hello"}],
    )
    create_mock.assert_called_once_with(
        model="grok-4.6",
        messages=[{"role": "user", "content": "hello"}],
    )


def test_custom_endpoint_qwen_uses_registry():
    config = SubtitleConfig(
        openai_base_url="https://example.com/v1",
        openai_api_key="test-key",
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


def test_custom_endpoint_qwen3_gguf_disables_thinking():
    config = SubtitleConfig(
        openai_base_url="https://example.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="Qwen3.8-27B-UD-Q3_K_XL",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="Qwen3.8-27B-UD-Q3_K_XL",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"enable_thinking": False},
    )


def test_unregistered_gpt_model_does_not_receive_reasoning_effort():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
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
    )


@pytest.mark.parametrize(
    "model",
    ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
)
def test_openai_gpt_5_6_family_disables_reasoning(model):
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="none",
    )


def test_deepseek_v4_pro_disables_thinking():
    config = SubtitleConfig(
        openai_base_url="https://api.deepseek.com",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_openrouter_deepseek_v4_uses_provider_encoding():
    config = SubtitleConfig(
        openai_base_url="https://openrouter.ai/api/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="deepseek/deepseek-v4-flash",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="deepseek/deepseek-v4-flash",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"reasoning": {"effort": "none"}},
    )


def test_openrouter_gpt_5_6_luna_disables_reasoning():
    config = SubtitleConfig(
        openai_base_url="https://openrouter.ai/api/v1",
        openai_api_key="test-key",
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


def test_unregistered_future_gpt_model_does_not_receive_reasoning_effort():
    config = SubtitleConfig(
        openai_base_url="https://example.com/v1",
        openai_api_key="test-key",
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
    )


def test_official_deepseek_disables_thinking_for_all_models():
    config = SubtitleConfig(
        openai_base_url="https://api.deepseek.com",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="deepseek-v4-other",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="deepseek-v4-other",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_custom_endpoint_unregistered_deepseek_variant_is_unchanged():
    config = SubtitleConfig(
        openai_base_url="https://example.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="deepseek-v4-other",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="deepseek-v4-other",
        messages=[{"role": "user", "content": "hello"}],
    )


def test_original_gpt_5_does_not_receive_reasoning_effort():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
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
    )


def test_disable_thinking_overrides_explicit_reasoning_options():
    config = SubtitleConfig(
        openai_base_url="https://openrouter.ai/api/v1",
        openai_api_key="test-key",
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


def test_response_usage_reads_top_level_reasoning_tokens():
    response = {
        "id": "chatcmpl-top-level",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
            "reasoning_tokens": 12,
        },
    }

    assert get_response_usage(response)["reasoning_tokens"] == 12


def test_response_usage_reads_output_tokens_details():
    response = SimpleNamespace(
        id="chatcmpl-output-details",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18,
            output_tokens_details=SimpleNamespace(reasoning_tokens=15),
        ),
    )

    assert get_response_usage(response)["reasoning_tokens"] == 15


def test_zero_reasoning_tokens_does_not_hide_later_fields():
    response = SimpleNamespace(
        id="chatcmpl-zero-first",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18,
            reasoning_tokens=9,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )

    assert get_response_usage(response)["reasoning_tokens"] == 9


def test_request_metrics_compute_throughput_from_known_usage_only():
    metrics = [
        RequestMetric(
            started_at=0.0,
            ended_at=10.0,
            latency=10.0,
            success=True,
            completion_tokens=100,
            finish_reason="stop",
            content_chars=300,
            error_type=None,
        ),
        RequestMetric(
            started_at=0.0,
            ended_at=100.0,
            latency=100.0,
            success=True,
            completion_tokens=None,
            finish_reason="stop",
            content_chars=50,
            error_type=None,
        ),
        RequestMetric(
            started_at=200.0,
            ended_at=260.0,
            latency=60.0,
            success=False,
            completion_tokens=None,
            finish_reason=None,
            content_chars=0,
            error_type="APITimeoutError",
        ),
    ]

    stats = summarize_request_metrics(metrics)

    assert stats["effective_tps"] == 10.0
    assert stats["throughput_requests"] == 1
    assert stats["missing_usage"] == 1
    assert stats["successful_requests"] == 2
    assert stats["failed_requests"] == 1
    assert stats["latency_avg"] == pytest.approx(170 / 3)
    assert stats["latency_max"] == 100.0


def test_nonempty_reasoning_content_warns_when_tokens_are_zero(monkeypatch):
    config = SubtitleConfig(
        openai_base_url="https://api.deepseek.com",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    response = _completion_response(reasoning_tokens=0)
    response.choices[0].message.reasoning_content = "internal chain of thought"
    client._client.chat.completions.create = Mock(return_value=response)
    terminal_notice = Mock()
    log_warning = Mock()
    monkeypatch.setattr(llm_client, "rich_print", terminal_notice)
    monkeypatch.setattr(llm_client.logger, "warning", log_warning)

    client.create_chat_completion(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hello"}],
    )

    terminal_notice.assert_called_once()
    notice = terminal_notice.call_args.args[0]
    assert "检测到模型实际使用了思考模式（关闭参数未生效）" in notice
    assert "reasoning_content=present" in notice
    log_warning.assert_called_once()


def test_create_chat_completion_observes_dict_shaped_response(monkeypatch):
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    response = {
        "id": "chatcmpl-dict",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "reasoning_tokens": 4,
        },
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "translated",
                    "reasoning_content": "thinking",
                },
            }
        ],
    }
    client._client.chat.completions.create = Mock(return_value=response)
    terminal_notice = Mock()
    monkeypatch.setattr(llm_client, "rich_print", terminal_notice)

    result = client.create_chat_completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result is response
    stats = client.metrics_summary()
    assert stats["successful_requests"] == 1
    assert stats["throughput_requests"] == 1
    assert stats["effective_tps"] is not None
    assert "reasoning_tokens=4" in terminal_notice.call_args.args[0]


def test_observation_errors_do_not_fail_successful_response(monkeypatch):
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    response = _completion_response(reasoning_tokens=0)
    client._client.chat.completions.create = Mock(return_value=response)
    monkeypatch.setattr(
        llm_client,
        "get_response_usage",
        Mock(side_effect=RuntimeError("usage boom")),
    )
    monkeypatch.setattr(
        llm_client.logger,
        "info",
        Mock(side_effect=RuntimeError("log boom")),
    )

    result = client.create_chat_completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result is response


def test_metric_lock_errors_do_not_fail_successful_response():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    response = _completion_response(reasoning_tokens=0)
    client._client.chat.completions.create = Mock(return_value=response)
    client._record_metric = Mock(side_effect=RuntimeError("lock boom"))

    result = client.create_chat_completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result is response


def test_failed_request_metric_errors_do_not_mask_api_error():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    client._client.chat.completions.create = Mock(side_effect=TimeoutError("timeout"))
    client._record_metric = Mock(side_effect=RuntimeError("lock boom"))

    with pytest.raises(TimeoutError, match="timeout"):
        client.create_chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_detected_reasoning_warns_once_under_concurrency(monkeypatch):
    config = SubtitleConfig(
        openai_base_url="https://api.deepseek.com",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    client._client.chat.completions.create = Mock(
        return_value=_completion_response(reasoning_tokens=88)
    )
    terminal_notice = Mock()
    log_warning = Mock()
    monkeypatch.setattr(llm_client, "rich_print", terminal_notice)
    monkeypatch.setattr(llm_client.logger, "warning", log_warning)

    def _request(_index: int) -> None:
        client.create_chat_completion(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "hello"}],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_request, range(16)))

    terminal_notice.assert_called_once()
    log_warning.assert_called_once()


def test_custom_endpoint_prefixed_deepseek_v4_disables_thinking():
    config = SubtitleConfig(
        openai_base_url="https://example.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    create_mock = Mock()
    client._client.chat.completions.create = create_mock

    client.create_chat_completion(
        model="vendor/deepseek-v4-flash",
        messages=[{"role": "user", "content": "hello"}],
    )

    create_mock.assert_called_once_with(
        model="vendor/deepseek-v4-flash",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"thinking": {"type": "disabled"}},
    )
