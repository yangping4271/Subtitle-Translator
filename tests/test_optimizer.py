from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core.optimizer import (
    SubtitleOptimizer,
    TRANSLATION_ONLY_RESPONSE_FORMAT,
    TRANSLATION_RESPONSE_FORMAT,
)


def _build_optimizer(base_url: str) -> SubtitleOptimizer:
    optimizer = object.__new__(SubtitleOptimizer)
    optimizer.config = SubtitleConfig(
        openai_base_url=base_url,
        _skip_env_load=True,
    )
    return optimizer


def test_optimizer_uses_translation_only_schema_for_local_endpoint():
    optimizer = _build_optimizer("http://127.0.0.1:1234/v1")

    assert optimizer._should_use_translation_only_schema() is True
    assert optimizer._get_required_response_fields() == "`id` and `translation`"
    assert optimizer._get_translation_response_format() == TRANSLATION_ONLY_RESPONSE_FORMAT


def test_optimizer_uses_full_schema_for_remote_endpoint():
    optimizer = _build_optimizer("https://api.openai.com/v1")

    assert optimizer._should_use_translation_only_schema() is False
    assert optimizer._get_required_response_fields() == "`id`, `optimized`, and `translation`"
    assert optimizer._get_translation_response_format() == TRANSLATION_RESPONSE_FORMAT
