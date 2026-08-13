from subtitle_translator.translation_core.thinking import (
    ThinkingDisableMethod,
    get_thinking_disable_spec,
    normalize_model_name,
    thinking_disable_applies,
)


def test_normalize_model_name_strips_vendor_prefix():
    assert normalize_model_name("OpenAI/GPT-5.6-Luna") == "gpt-5.6-luna"
    assert normalize_model_name("deepseek/deepseek-v4-pro") == "deepseek-v4-pro"


def test_registered_gpt_5_6_models_use_openai_none():
    for model in (
        "gpt-5.6",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "openai/gpt-5.6-luna",
    ):
        spec = get_thinking_disable_spec(model)
        assert spec is not None
        assert spec.method is ThinkingDisableMethod.OPENAI_REASONING_EFFORT
        assert spec.reasoning_effort == "none"


def test_registered_deepseek_v4_models_use_thinking_disabled():
    for model in ("deepseek-v4-flash", "vendor/deepseek-v4-pro"):
        spec = get_thinking_disable_spec(model)
        assert spec is not None
        assert spec.method is ThinkingDisableMethod.DEEPSEEK_THINKING


def test_unregistered_models_are_not_in_the_table():
    assert get_thinking_disable_spec("gpt-5.1") is None
    assert get_thinking_disable_spec("gpt-4o") is None
    assert get_thinking_disable_spec("qwen-plus") is None
    assert get_thinking_disable_spec("deepseek-v4-other") is None
    assert get_thinking_disable_spec("gpt-6") is None
    assert get_thinking_disable_spec("gpt-5.6-pro") is None
    assert get_thinking_disable_spec("deepseek-v4-xxx") is None


def test_thinking_disable_applies_only_registered_models():
    assert thinking_disable_applies("gpt-5.6-luna")
    assert thinking_disable_applies("vendor/deepseek-v4-flash")
    assert not thinking_disable_applies("anthropic/claude-sonnet", "openrouter")
    assert not thinking_disable_applies("qwen-plus", "dashscope")
    assert not thinking_disable_applies("gpt-5.1")
    assert not thinking_disable_applies("qwen-plus")
