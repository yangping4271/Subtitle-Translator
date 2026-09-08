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


def _metric(**fields):
    return RequestMetric(
        **{"success": True, "finish_reason": "stop", "error_type": None, **fields}
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


def test_cannot_disable_grok_reports_default_reasoning(monkeypatch):
    config = SubtitleConfig(
        openai_base_url="https://api.x.ai/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    client._client.chat.completions.create = Mock(
        return_value=_completion_response(reasoning_tokens=683)
    )
    terminal_notice = Mock()
    log_info = Mock()
    log_warning = Mock()
    monkeypatch.setattr(llm_client, "rich_print", terminal_notice)
    monkeypatch.setattr(llm_client.logger, "info", log_info)
    monkeypatch.setattr(llm_client.logger, "warning", log_warning)

    client.create_chat_completion(
        model="grok-4.6",
        messages=[{"role": "user", "content": "hello"}],
    )

    notice = terminal_notice.call_args.args[0]
    assert "该模型无法关闭思考，使用默认强度" in notice
    assert "grok-4.6" in notice
    assert "reasoning_tokens=683" in notice
    assert "最低强度" not in notice
    log_warning.assert_not_called()
    assert any(
        "该模型无法关闭思考，使用默认强度: grok-4.6, reasoning_tokens=683"
        in str(call.args[0])
        for call in log_info.call_args_list
    )


def test_unknown_model_reports_default_reasoning(monkeypatch):
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    client._client.chat.completions.create = Mock(
        return_value=_completion_response(reasoning_tokens=12)
    )
    terminal_notice = Mock()
    monkeypatch.setattr(llm_client, "rich_print", terminal_notice)

    client.create_chat_completion(
        model="gpt-5",
        messages=[{"role": "user", "content": "hello"}],
    )

    notice = terminal_notice.call_args.args[0]
    assert "该模型无法关闭思考，使用默认强度" in notice
    assert "gpt-5" in notice
    assert "reasoning_tokens=12" in notice
    assert "最低强度" not in notice


def test_cannot_disable_model_reports_default_reasoning(monkeypatch):
    config = SubtitleConfig(
        openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    client._client.chat.completions.create = Mock(
        return_value=_completion_response(reasoning_tokens=9)
    )
    terminal_notice = Mock()
    monkeypatch.setattr(llm_client, "rich_print", terminal_notice)

    client.create_chat_completion(
        model="gemini-2.5-pro",
        messages=[{"role": "user", "content": "hello"}],
    )

    notice = terminal_notice.call_args.args[0]
    assert "该模型无法关闭思考，使用默认强度" in notice
    assert "gemini-2.5-pro" in notice
    assert "reasoning_tokens=9" in notice
    assert "最低强度" not in notice


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
        _metric(
            started_at=0.0,
            ended_at=10.0,
            latency=10.0,
            completion_tokens=100,
            content_chars=300,
        ),
        _metric(
            started_at=0.0,
            ended_at=20.0,
            latency=20.0,
            completion_tokens=200,
            content_chars=600,
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
        _metric(
            started_at=0.0,
            ended_at=5.0,
            latency=5.0,
            completion_tokens=274,
            content_chars=1158,
            prompt_tokens=865,
            total_tokens=1139,
        ),
        _metric(
            started_at=5.0,
            ended_at=22.0,
            latency=17.0,
            completion_tokens=1180,
            content_chars=3685,
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
        _metric(
            started_at=0.0,
            ended_at=31.0,
            latency=31.0,
            completion_tokens=None,
            finish_reason="length",
            content_chars=0,
        ),
        _metric(
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
        _metric(
            started_at=0.0,
            ended_at=10.0,
            latency=10.0,
            completion_tokens=100,
            content_chars=300,
        ),
        _metric(
            started_at=0.0,
            ended_at=100.0,
            latency=100.0,
            completion_tokens=None,
            content_chars=50,
        ),
        _metric(
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


@pytest.mark.parametrize(
    "base_url, model, disable_thinking, options, expected",
    [
        ("https://example.com/v1", "gemma-4-e4b-it", True, {}, {}),
        (
            "https://api.deepseek.com",
            "deepseek-v4-flash",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        (
            "https://openrouter.ai/api/v1",
            "google/gemini-3-flash-preview",
            True,
            {},
            {"extra_body": {"reasoning": {"effort": "none"}}},
        ),
        (
            "https://ai-proxy.chatwise.app/openrouter/api/v1",
            "qwen/qwen3.6-27b",
            True,
            {},
            {"extra_body": {"reasoning": {"effort": "none"}}},
        ),
        (
            "https://open.bigmodel.cn/api/paas/v4/",
            "glm-4-flash",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        (
            "https://api.minimax.io/v1",
            "MiniMax-M3",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        (
            "https://api.moonshot.cn/v1",
            "kimi-k2.6",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        ("https://api.moonshot.cn/v1", "kimi-k3", True, {}, {}),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "gemini-3.6-flash",
            True,
            {},
            {
                "extra_body": {
                    "extra_body": {
                        "google": {"thinking_config": {"thinking_level": "minimal"}}
                    }
                }
            },
        ),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "gemini-2.5-flash",
            True,
            {},
            {
                "extra_body": {
                    "extra_body": {
                        "google": {"thinking_config": {"thinking_budget": 0}}
                    }
                }
            },
        ),
        ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", True, {}, {}),
        ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", True, {}, {}),
        (
            "https://example.com/v1",
            "glm-5.2",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen-vl-plus",
            True,
            {},
            {"extra_body": {"enable_thinking": False}},
        ),
        (
            "https://ark.cn-beijing.volces.com/api/v3",
            "ep-unregistered",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        (
            "https://api.anthropic.com/v1/",
            "claude-sonnet-5",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        ("https://api.anthropic.com/v1/", "claude-fable-5", True, {}, {}),
        ("https://api.x.ai/v1", "grok-4.3", True, {}, {"reasoning_effort": "none"}),
        ("https://api.x.ai/v1", "grok-4.6", True, {}, {}),
        ("https://example.com/v1", "glm-5.3-flash", True, {}, {}),
        ("https://open.bigmodel.cn/api/paas/v4/", "glm-5.3-flash", True, {}, {}),
        (
            "https://example.com/v1",
            "qwen-plus",
            True,
            {},
            {"extra_body": {"enable_thinking": False}},
        ),
        (
            "https://example.com/v1",
            "Qwen3.8-27B-UD-Q3_K_XL",
            True,
            {},
            {"extra_body": {"enable_thinking": False}},
        ),
        ("https://api.openai.com/v1", "gpt-5.1", True, {}, {}),
        (
            "https://api.openai.com/v1",
            "gpt-5.6",
            True,
            {},
            {"reasoning_effort": "none"},
        ),
        (
            "https://api.openai.com/v1",
            "gpt-5.6-sol",
            True,
            {},
            {"reasoning_effort": "none"},
        ),
        (
            "https://api.openai.com/v1",
            "gpt-5.6-terra",
            True,
            {},
            {"reasoning_effort": "none"},
        ),
        (
            "https://api.openai.com/v1",
            "gpt-5.6-luna",
            True,
            {},
            {"reasoning_effort": "none"},
        ),
        (
            "https://api.deepseek.com",
            "deepseek-v4-pro",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        (
            "https://openrouter.ai/api/v1",
            "deepseek/deepseek-v4-flash",
            True,
            {},
            {"extra_body": {"reasoning": {"effort": "none"}}},
        ),
        (
            "https://openrouter.ai/api/v1",
            "openai/gpt-5.6-luna",
            True,
            {},
            {"extra_body": {"reasoning": {"effort": "none"}}},
        ),
        ("https://example.com/v1", "openai/gpt-6", True, {}, {}),
        (
            "https://api.deepseek.com",
            "deepseek-v4-other",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        (
            "https://example.com/v1",
            "deepseek-v4-other",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        (
            "https://example.com/v1",
            "deepseek-v4.1-flash-expires-on-0910",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        ("https://api.openai.com/v1", "gpt-5", True, {}, {}),
        (
            "https://openrouter.ai/api/v1",
            "openai/gpt-5.6-luna",
            True,
            {
                "reasoning_effort": "high",
                "extra_body": {"reasoning": {"effort": "high"}},
            },
            {"extra_body": {"reasoning": {"effort": "none"}}},
        ),
        (
            "https://example.com/codex/v1",
            "gpt-5.6-luna",
            True,
            {},
            {"reasoning_effort": "none"},
        ),
        ("https://api.openai.com/v1", "gpt-4o", True, {}, {}),
        ("https://api.deepseek.com", "deepseek-v4-flash", False, {}, {}),
        ("https://api.openai.com/v1", "gpt-5.6-luna", False, {}, {}),
        (
            "https://example.com/v1",
            "vendor/deepseek-v4-flash",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        (
            "https://example.com/v1",
            "deepseek-v4-flash-0731",
            True,
            {},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
    ],
)
def test_provider_request_options(base_url, model, disable_thinking, options, expected):
    client = LLMClient(
        SubtitleConfig(
            openai_base_url=base_url,
            openai_api_key="test-key",
            disable_thinking=disable_thinking,
        )
    )
    create = Mock(return_value=_completion_response(reasoning_tokens=0))
    client._client.chat.completions.create = create
    messages = [{"role": "user", "content": "hello"}]
    try:
        client.create_chat_completion(model=model, messages=messages, **options)
        create.assert_called_once_with(model=model, messages=messages, **expected)
    finally:
        client.close()
