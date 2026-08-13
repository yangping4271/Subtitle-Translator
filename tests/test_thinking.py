from subtitle_translator.translation_core.thinking import (
    ThinkingDisableMethod,
    encode_thinking_extra_body,
    get_provider_thinking_method,
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
        assert spec.method is ThinkingDisableMethod.THINKING_TYPE_DISABLED


def test_registered_mainstream_models_use_expected_methods():
    for model in ("glm-5.2", "kimi-k2.6", "MiniMax-M3"):
        spec = get_thinking_disable_spec(model)
        assert spec is not None
        assert spec.method is ThinkingDisableMethod.THINKING_TYPE_DISABLED

    for model in (
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "google/gemini-3-flash-preview",
    ):
        spec = get_thinking_disable_spec(model)
        assert spec is not None
        assert spec.method is ThinkingDisableMethod.GOOGLE_THINKING_LEVEL

    spec = get_thinking_disable_spec("google/gemini-2.5-flash")
    assert spec is not None
    assert spec.method is ThinkingDisableMethod.GOOGLE_THINKING_BUDGET

    for model in ("claude-sonnet-5", "anthropic/claude-opus-4-6"):
        spec = get_thinking_disable_spec(model)
        assert spec is not None
        assert spec.method is ThinkingDisableMethod.THINKING_TYPE_DISABLED

    spec = get_thinking_disable_spec("x-ai/grok-4.3")
    assert spec is not None
    assert spec.method is ThinkingDisableMethod.OPENAI_REASONING_EFFORT
    assert spec.reasoning_effort == "none"

    spec = get_thinking_disable_spec("qwen-plus")
    assert spec is not None
    assert spec.method is ThinkingDisableMethod.ENABLE_THINKING_FALSE

    spec = get_thinking_disable_spec("doubao-seed-1-6")
    assert spec is not None
    assert spec.method is ThinkingDisableMethod.THINKING_TYPE_DISABLED


def test_models_that_cannot_disable_thinking_are_not_registered():
    assert get_thinking_disable_spec("kimi-k3") is None
    assert get_thinking_disable_spec("kimi-k2.7-code") is None
    assert get_thinking_disable_spec("minimax-m2.7") is None
    assert get_thinking_disable_spec("claude-fable-5") is None
    assert get_thinking_disable_spec("claude-mythos-5") is None
    assert get_thinking_disable_spec("grok-4.6") is None
    assert get_thinking_disable_spec("grok-4.5") is None
    assert get_thinking_disable_spec("qwq-plus") is None
    assert get_thinking_disable_spec("gpt-oss-120b") is None
    assert get_thinking_disable_spec("qwen3-32b") is None


def test_unregistered_models_are_not_in_the_table():
    assert get_thinking_disable_spec("gpt-5.1") is None
    assert get_thinking_disable_spec("gpt-4o") is None
    assert get_thinking_disable_spec("deepseek-v4-other") is None
    assert get_thinking_disable_spec("gpt-6") is None
    assert get_thinking_disable_spec("gpt-5.6-pro") is None
    assert get_thinking_disable_spec("deepseek-v4-xxx") is None


def test_thinking_disable_applies_registry_or_official_providers():
    assert thinking_disable_applies("gpt-5.6-luna")
    assert thinking_disable_applies("vendor/deepseek-v4-flash")
    assert thinking_disable_applies("anthropic/claude-sonnet", "openrouter")
    assert thinking_disable_applies("any-model", "deepseek")
    assert thinking_disable_applies("any-model", "zhipu")
    assert thinking_disable_applies("any-model", "minimax")
    assert thinking_disable_applies("any-model", "dashscope")
    assert thinking_disable_applies("any-model", "volcengine")
    assert thinking_disable_applies("qwen-plus")
    assert not thinking_disable_applies("gpt-5.1", "openai")
    assert not thinking_disable_applies("kimi-k3", "kimi")
    assert not thinking_disable_applies("llama-3.3-70b", "groq")
    assert not thinking_disable_applies("claude-fable-5", "anthropic")
    assert not thinking_disable_applies("grok-4.6", "xai")


def test_google_thinking_config_encodes_one_field_per_generation():
    gemini_3 = encode_thinking_extra_body(
        {}, ThinkingDisableMethod.GOOGLE_THINKING_LEVEL
    )
    gemini_25 = encode_thinking_extra_body(
        {}, ThinkingDisableMethod.GOOGLE_THINKING_BUDGET
    )

    assert gemini_3["extra_body"]["google"]["thinking_config"] == {
        "thinking_level": "minimal",
    }
    assert gemini_25["extra_body"]["google"]["thinking_config"] == {
        "thinking_budget": 0,
    }
    assert encode_thinking_extra_body(
        {}, ThinkingDisableMethod.ENABLE_THINKING_FALSE
    ) == {"enable_thinking": False}


def test_provider_thinking_method_only_for_unified_official_switches():
    assert (
        get_provider_thinking_method("openrouter")
        is ThinkingDisableMethod.OPENROUTER_REASONING
    )
    assert (
        get_provider_thinking_method("deepseek")
        is ThinkingDisableMethod.THINKING_TYPE_DISABLED
    )
    assert (
        get_provider_thinking_method("dashscope")
        is ThinkingDisableMethod.ENABLE_THINKING_FALSE
    )
    assert (
        get_provider_thinking_method("volcengine")
        is ThinkingDisableMethod.THINKING_TYPE_DISABLED
    )
    assert get_provider_thinking_method("openai") is None
    assert get_provider_thinking_method("kimi") is None
    assert get_provider_thinking_method("google") is None
    assert get_provider_thinking_method("groq") is None
    assert get_provider_thinking_method("anthropic") is None
    assert get_provider_thinking_method("xai") is None
