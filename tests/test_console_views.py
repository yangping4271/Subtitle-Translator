from subtitle_translator.console_views import show_model_config


def test_show_model_config_shows_reasoning_disabled(capsys):
    show_model_config("gpt-5.6-luna", "openai/gpt-6")

    output = capsys.readouterr().out
    assert "断句: gpt-5.6-luna (思考模式: 已关闭)" in output
    assert "翻译: openai/gpt-6 (思考模式: 已关闭)" in output


def test_show_model_config_explains_original_gpt_5_limit(capsys):
    show_model_config("gpt-5", "gpt-5-mini")

    output = capsys.readouterr().out
    assert "断句: gpt-5 (推理强度: minimal，模型不支持关闭)" in output
    assert "翻译: gpt-5-mini (推理强度: minimal，模型不支持关闭)" in output


def test_show_model_config_shows_provider_level_disable(capsys):
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
