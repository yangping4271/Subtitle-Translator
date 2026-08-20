from subtitle_translator.console_views import (
    show_api_performance_stats,
    show_model_config,
    show_time_stats,
)


def test_show_model_config_shows_reasoning_disabled(capsys):
    show_model_config("gpt-5.6-luna", "openai/gpt-5.6-sol")

    output = capsys.readouterr().out
    assert "断句: gpt-5.6-luna (思考模式: 已关闭)" in output
    assert "翻译: openai/gpt-5.6-sol (思考模式: 已关闭)" in output


def test_show_model_config_omits_reasoning_for_unregistered_models(capsys):
    show_model_config("gpt-5", "gpt-5.1")

    output = capsys.readouterr().out
    assert "思考模式" not in output
    assert "推理强度" not in output


def test_show_model_config_shows_official_provider_disable(capsys):
    show_model_config("glm-4-flash", "any-model", provider_type="zhipu")

    output = capsys.readouterr().out
    assert "断句: glm-4-flash (思考模式: 已关闭)" in output
    assert "翻译: any-model (思考模式: 已关闭)" in output


def test_show_model_config_shows_openrouter_provider_disable(capsys):
    show_model_config(
        "qwen/qwen3.6-27b",
        "anthropic/claude-sonnet",
        provider_type="openrouter",
    )

    output = capsys.readouterr().out
    assert "断句: qwen/qwen3.6-27b (思考模式: 已关闭)" in output
    assert "翻译: anthropic/claude-sonnet (思考模式: 已关闭)" in output


def test_show_model_config_omits_reasoning_for_unsupported_model(capsys):
    show_model_config("gpt-4o-mini", "gpt-4o")

    output = capsys.readouterr().out
    assert "推理强度" not in output
    assert "思考模式" not in output


def test_show_api_performance_stats_uses_merged_request_view(capsys):
    show_api_performance_stats(
        {
            "requests": 4,
            "successful_requests": 4,
            "failed_requests": 0,
            "latency_avg": 31.8,
            "latency_p95": 50.8,
            "latency_max": 53.8,
            "effective_tps": 40.1,
            "slow_requests": 3,
            "slowest_request": 53.8,
            "anomalies": 0,
            "empty_responses": 0,
            "abnormal_finishes": 0,
            "unknown_finishes": 0,
            "missing_usage": 0,
        }
    )

    output = capsys.readouterr().out
    assert "API 性能统计" in output
    assert "请求: 4 次 (成功 4 / 失败 0)" in output
    assert "平均 31.8s，P95 50.8s，最大 53.8s" in output
    assert "有效吞吐: 40.1 token/s" in output
    assert "最长上下文: 无统计" in output
    assert "慢请求: 3 次 (>30s，最慢 53.8s)" in output
    assert "响应异常: 0 次" in output
    assert "TTFT" not in output
    assert "输入 Token" not in output


def test_show_api_performance_stats_shows_throughput_coverage(capsys):
    show_api_performance_stats(
        {
            "requests": 3,
            "successful_requests": 2,
            "failed_requests": 1,
            "latency_avg": 12.0,
            "latency_p95": 20.0,
            "latency_max": 21.0,
            "effective_tps": 10.0,
            "throughput_requests": 1,
            "slow_requests": 0,
            "slowest_request": 21.0,
            "anomalies": 2,
            "empty_responses": 1,
            "abnormal_finishes": 1,
            "unknown_finishes": 0,
            "missing_usage": 1,
        }
    )

    output = capsys.readouterr().out
    assert "基于 1/2 个成功请求，另 1 次缺少统计" in output
    assert (
        "响应异常: 2 次 (同一响应可重复命中：空响应 1，异常结束 1，响应统计缺失 1)"
        in output
    )


def test_show_api_performance_stats_shows_longest_context(capsys):
    show_api_performance_stats(
        {
            "requests": 4,
            "successful_requests": 4,
            "failed_requests": 0,
            "latency_avg": 10.0,
            "latency_p95": 16.0,
            "latency_max": 17.0,
            "effective_tps": 60.0,
            "slow_requests": 0,
            "slowest_request": 17.0,
            "anomalies": 0,
            "empty_responses": 0,
            "abnormal_finishes": 0,
            "unknown_finishes": 0,
            "missing_usage": 0,
            "max_context_tokens": 2591,
            "context_reference_tokens": 4096,
            "max_context_ratio": 2591 / 4096,
        }
    )

    output = capsys.readouterr().out
    assert "最长上下文: 2591 token（4K 的 63%）" in output


def test_show_model_config_shows_prefixed_deepseek_as_disabled(capsys):
    show_model_config("vendor/deepseek-v4-flash", "deepseek-v4-pro")

    output = capsys.readouterr().out
    assert "断句: vendor/deepseek-v4-flash (思考模式: 已关闭)" in output
    assert "翻译: deepseek-v4-pro (思考模式: 已关闭)" in output


def test_show_time_stats_hides_values_that_round_to_zero(capsys):
    show_time_stats(
        {
            "📋 上下文提取": 0.01,
            "✂️ 智能断句": 54.4,
            "🌍 批量翻译": 34.0,
            "💾 保存字幕": 0.049,
        },
        total_time=88.5,
    )

    output = capsys.readouterr().out
    assert "上下文提取" not in output
    assert "保存字幕" not in output
    assert "智能断句: 54.4s" in output
    assert "批量翻译: 34.0s" in output
    assert "总计: 88.5s" in output
