from subtitle_translator.console_views import show_model_config


def test_show_model_config_includes_reasoning_effort(capsys):
    show_model_config("gpt-5.6-luna", "openai/gpt-6")

    output = capsys.readouterr().out
    assert "断句: gpt-5.6-luna (推理强度: low)" in output
    assert "翻译: openai/gpt-6 (推理强度: low)" in output


def test_show_model_config_omits_reasoning_for_unsupported_model(capsys):
    show_model_config("gpt-4o-mini", "gpt-4o")

    output = capsys.readouterr().out
    assert "推理强度" not in output
