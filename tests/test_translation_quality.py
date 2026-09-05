import json
from types import SimpleNamespace

from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core.data import SubtitleData, SubtitleSegment
from subtitle_translator.translation_core.translation_execution import (
    TRANSLATION_JSON_OBJECT_RESPONSE_FORMAT,
    TRANSLATION_RESPONSE_FORMAT,
    TranslationEngine,
)
from subtitle_translator.translation_core.translation_context import TranslationContext


DEFAULT_TRANSLATION_CONTEXT = TranslationContext(target_language="简体中文")


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
    )
    adapter = CapturingAdapter(optimized)
    translation_batch = SubtitleData(
        [SubtitleSegment("Music.", start_time=0, end_time=1000)]
    )

    with TranslationEngine(config, adapter, DEFAULT_TRANSLATION_CONTEXT) as engine:
        results = engine.translate_batch(translation_batch, context_info="course intro")

    return adapter.requests[0], results


def test_translation_engine_uses_full_schema_for_remote_endpoint():
    request, _ = _translate_once("https://api.openai.com/v1")

    assert request["response_format"] == TRANSLATION_RESPONSE_FORMAT
    example = request["messages"][0]["content"].splitlines()
    example = next(line for line in example if line.startswith('{"subtitles":['))
    assert set(json.loads(example)["subtitles"][0]) == {
        "id", "optimized", "translation", "discarded"
    }


def test_translation_engine_uses_json_object_for_deepseek_endpoint():
    request, _ = _translate_once("https://api.deepseek.com/v1")

    assert request["response_format"] == TRANSLATION_JSON_OBJECT_RESPONSE_FORMAT


def test_translation_prompt_explicitly_requests_json_output():
    request, _ = _translate_once("https://api.openai.com/v1")

    assert "json" in request["messages"][0]["content"].lower()
    assert "code fences" in request["messages"][0]["content"].lower()


def test_translation_engine_allows_term_corrections():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=1,
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

    with TranslationEngine(config, adapter, DEFAULT_TRANSLATION_CONTEXT) as engine:
        results = engine.translate_batch(translation_batch, context_info="")

    assert results[0]["optimized"] == (
        "using a database, LangChain, and an LM-powered pipeline"
    )


def test_shifted_source_invalidates_translation_and_discard():
    original = "extraction, and tool retrieval."
    shifted = "These architectures address agents failing at long horizon tasks."
    config = SubtitleConfig(thread_num=1)
    with TranslationEngine(config, CapturingAdapter(), DEFAULT_TRANSLATION_CONTEXT) as engine:
        results = engine._build_translation_results(
            {"1": {"optimized_subtitle": shifted, "translation": "不相关的译文", "discarded": True}},
            {"1": original},
        )
    assert results[0]["optimized"] == original
    assert results[0]["translation"] == ""
    assert results[0]["discarded"] is False


def test_shifted_item_retries_without_retranslating_successful_items():
    class Adapter:
        def __init__(self):
            self.requests = []

        def create_chat_completion(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                items = [
                    {"id": 1, "optimized": "We use two methods:", "translation": "我们使用两种方法："},
                    {"id": 2, "optimized": "These architectures solve unrelated problems.", "translation": "错误译文"},
                    {"id": 3, "optimized": "Both improve accuracy.", "translation": "两者都能提高准确率。"},
                ]
            else:
                items = [{"id": 2, "optimized": "extraction, and tool retrieval.", "translation": "提取和工具检索。"}]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"subtitles": items})
            ))])

    adapter = Adapter()
    source = SubtitleData([
        SubtitleSegment(text, i * 1000, (i + 1) * 1000)
        for i, text in enumerate([
            "We use two methods:", "extraction, and tool retrieval.", "Both improve accuracy."
        ])
    ])
    with TranslationEngine(SubtitleConfig(thread_num=1), adapter, DEFAULT_TRANSLATION_CONTEXT) as engine:
        results = engine.translate_batch(source, "Technical lecture")
    assert len(adapter.requests) == 2
    assert [r["translation"] for r in results] == [
        "我们使用两种方法：", "提取和工具检索。", "两者都能提高准确率。"
    ]
    retry_input = adapter.requests[1]["messages"][-1]["content"]
    payload = json.loads(retry_input.split("<subtitles>")[1].split("</subtitles>")[0])
    assert payload == {"2": "extraction, and tool retrieval."}
    assert "Technical lecture" in retry_input
    assert "We use two methods:" not in retry_input
    assert "Both improve accuracy." not in retry_input
    assert "错误译文" not in retry_input
    assert "我们使用两种方法" not in retry_input
