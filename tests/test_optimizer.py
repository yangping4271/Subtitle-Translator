from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core.optimizer import (
    SubtitleOptimizer,
    TRANSLATION_JSON_OBJECT_RESPONSE_FORMAT,
    TRANSLATION_ONLY_RESPONSE_FORMAT,
    TRANSLATION_RESPONSE_FORMAT,
    _is_suspicious_optimized_shift,
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
    assert optimizer._get_required_response_fields() == "`id`, `translation`, and `discarded`"
    assert optimizer._get_translation_response_format() == TRANSLATION_ONLY_RESPONSE_FORMAT


def test_optimizer_uses_full_schema_for_remote_endpoint():
    optimizer = _build_optimizer("https://api.openai.com/v1")

    assert optimizer._should_use_translation_only_schema() is False
    assert optimizer._get_required_response_fields() == "`id`, `optimized`, `translation`, and `discarded`"
    assert optimizer._get_translation_response_format() == TRANSLATION_RESPONSE_FORMAT


def test_optimizer_uses_json_object_for_deepseek_endpoint():
    optimizer = _build_optimizer("https://api.deepseek.com/v1")

    assert optimizer._should_use_translation_only_schema() is False
    assert optimizer._get_required_response_fields() == "`id`, `optimized`, `translation`, and `discarded`"
    assert optimizer._get_translation_response_format() == TRANSLATION_JSON_OBJECT_RESPONSE_FORMAT


def test_translate_batch_directly_passes_real_batch_info():
    optimizer = object.__new__(SubtitleOptimizer)
    optimizer.batch_logs = []

    captured = {}

    def fake_translate(subtitle_json, context_info, batch_num=None, total_batches=None):
        captured["translate"] = {
            "subtitle_json": subtitle_json,
            "context_info": context_info,
            "batch_num": batch_num,
            "total_batches": total_batches,
        }
        return [{"id": 1, "original": "hello", "optimized": "hello", "translation": "你好"}]

    class DummyExecutor:
        def retry_failed_translations(self, failed_items, context_info, results, batch_num=None, total_batches=None):
            captured["retry"] = {
                "failed_items": failed_items,
                "context_info": context_info,
                "results": results,
                "batch_num": batch_num,
                "total_batches": total_batches,
            }
            return results

    class DummyAsrData:
        def to_json(self):
            return {1: {"original_subtitle": "hello"}}

    optimizer._translate = fake_translate
    optimizer._executor_obj = DummyExecutor()

    result = optimizer.translate_batch_directly(
        DummyAsrData(),
        "ctx",
        batch_num=3,
        total_batches=9,
    )

    assert result == [{"id": 1, "original": "hello", "optimized": "hello", "translation": "你好"}]
    assert captured["translate"]["batch_num"] == 3
    assert captured["translate"]["total_batches"] == 9
    assert "retry" not in captured


def test_create_translate_message_explicitly_requests_json_output():
    optimizer = _build_optimizer("https://api.openai.com/v1")

    class DummyExecutor:
        @staticmethod
        def _format_terminology(source_text=""):
            return ""

    optimizer._executor_obj = DummyExecutor()

    message = optimizer._create_translate_message({"1": "Music."}, "course intro")

    assert "json" in message[0]["content"].lower()
    assert "json" in message[1]["content"].lower()
    assert "code fences" in message[1]["content"].lower()


def test_translate_batch_directly_skips_retry_for_discarded_items():
    optimizer = object.__new__(SubtitleOptimizer)
    optimizer.batch_logs = []

    captured = {}

    def fake_translate(subtitle_json, context_info, batch_num=None, total_batches=None):
        return [{
            "id": 48,
            "original": "Music.",
            "optimized": "",
            "translation": "",
            "discarded": True,
        }]

    class DummyExecutor:
        def retry_failed_translations(self, failed_items, context_info, results, batch_num=None, total_batches=None):
            captured["retry"] = True
            return results

    class DummyAsrData:
        def to_json(self):
            return {48: {"original_subtitle": "Music."}}

    optimizer._translate = fake_translate
    optimizer._executor_obj = DummyExecutor()

    result = optimizer.translate_batch_directly(DummyAsrData(), "ctx")

    assert result[0]["discarded"] is True
    assert "retry" not in captured


def test_suspicious_optimized_shift_allows_term_corrections():
    assert _is_suspicious_optimized_shift(
        "using a database, land chain, and an LM-powered pipeline",
        "using a database, LangChain, and an LM-powered pipeline",
    ) is False


def test_suspicious_optimized_shift_detects_cross_id_move():
    assert _is_suspicious_optimized_shift(
        "extraction, and tool retrieval.",
        "With these, you learn how memory-first architectures address the problem of agents failing at long horizon tasks.",
    ) is True


def test_fill_missing_fields_reverts_suspicious_optimized_shift():
    optimizer = _build_optimizer("https://api.openai.com/v1")
    response_content = {
        "14": {
            "optimized_subtitle": (
                "With these, you learn how memory-first architectures address "
                "the problem of agents failing at long horizon tasks."
            ),
            "translation": "通过这些技术，您将了解记忆优先架构如何解决问题。",
            "discarded": False,
        }
    }
    original_subtitle = {"14": "extraction, and tool retrieval."}

    filled = optimizer._fill_missing_fields(response_content, original_subtitle)

    assert filled["14"]["optimized_subtitle"] == "extraction, and tool retrieval."
    assert filled["14"]["translation"] == "通过这些技术，您将了解记忆优先架构如何解决问题。"
