import json
from types import SimpleNamespace

from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core.data import SubtitleData, SubtitleSegment
from subtitle_translator.translation_core.translation_execution import (
    TRANSLATION_JSON_OBJECT_RESPONSE_FORMAT,
    TRANSLATION_ONLY_RESPONSE_FORMAT,
    TRANSLATION_RESPONSE_FORMAT,
    TranslationEngine,
)


class CapturingAdapter:
    def __init__(self, optimized: str = "Music."):
        self.optimized = optimized
        self.requests = []

    def create_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        content = json.dumps(
            {
                "subtitles": [
                    {
                        "id": 1,
                        "optimized": self.optimized,
                        "translation": "音乐",
                        "discarded": False,
                    }
                ]
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _translate_once(base_url: str, optimized: str = "Music."):
    config = SubtitleConfig(
        openai_base_url=base_url,
        thread_num=1,
        _skip_env_load=True,
    )
    adapter = CapturingAdapter(optimized)
    translation_batch = SubtitleData(
        [SubtitleSegment("Music.", start_time=0, end_time=1000)]
    )

    with TranslationEngine(config, adapter) as engine:
        results = engine.translate_batch(translation_batch, context_info="course intro")

    return adapter.requests[0], results


def test_translation_engine_uses_translation_only_schema_for_local_endpoint():
    request, _ = _translate_once("http://127.0.0.1:1234/v1")

    assert request["response_format"] == TRANSLATION_ONLY_RESPONSE_FORMAT
    assert "`id`, `translation`, and `discarded`" in request["messages"][0]["content"]


def test_translation_engine_uses_full_schema_for_remote_endpoint():
    request, _ = _translate_once("https://api.openai.com/v1")

    assert request["response_format"] == TRANSLATION_RESPONSE_FORMAT
    assert (
        "`id`, `optimized`, `translation`, and `discarded`"
        in request["messages"][0]["content"]
    )


def test_translation_engine_uses_json_object_for_deepseek_endpoint():
    request, _ = _translate_once("https://api.deepseek.com/v1")

    assert request["response_format"] == TRANSLATION_JSON_OBJECT_RESPONSE_FORMAT


def test_translation_prompt_explicitly_requests_json_output():
    request, _ = _translate_once("https://api.openai.com/v1")

    assert "json" in request["messages"][0]["content"].lower()
    assert "json" in request["messages"][1]["content"].lower()
    assert "code fences" in request["messages"][1]["content"].lower()


def test_translation_engine_allows_term_corrections():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=1,
        _skip_env_load=True,
    )
    adapter = CapturingAdapter(
        "using a database, LangChain, and an LM-powered pipeline"
    )
    translation_batch = SubtitleData(
        [
            SubtitleSegment(
                "using a database, land chain, and an LM-powered pipeline",
                start_time=0,
                end_time=1000,
            )
        ]
    )

    with TranslationEngine(config, adapter) as engine:
        results = engine.translate_batch(translation_batch, context_info="")

    assert results[0]["optimized"] == (
        "using a database, LangChain, and an LM-powered pipeline"
    )


def test_translation_engine_reverts_cross_id_optimized_shift():
    original = "extraction, and tool retrieval."
    shifted = (
        "With these, you learn how memory-first architectures address "
        "the problem of agents failing at long horizon tasks."
    )
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=1,
        _skip_env_load=True,
    )
    adapter = CapturingAdapter(shifted)
    translation_batch = SubtitleData(
        [SubtitleSegment(original, start_time=0, end_time=1000)]
    )

    with TranslationEngine(config, adapter) as engine:
        results = engine.translate_batch(translation_batch, context_info="")

    assert results[0]["optimized"] == original
    assert results[0]["translation"] == "音乐"
