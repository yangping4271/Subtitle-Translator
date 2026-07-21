import json
from types import SimpleNamespace

import pytest

from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core.data import SubtitleData, SubtitleSegment
from subtitle_translator.translation_core.translation_execution import TranslationEngine
from subtitle_translator import processor
from subtitle_translator.service import SubtitleTranslatorService


class StubModelAdapter:
    def __init__(self, responses: list[str | Exception]):
        self._responses = iter(responses)

    def create_chat_completion(self, **kwargs):
        content = next(self._responses)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_translation_batch_returns_structured_results_through_injected_model():
    response = json.dumps(
        {
            "subtitles": [
                {
                    "id": 1,
                    "optimized": "Hello world!",
                    "translation": "你好，世界！",
                    "discarded": False,
                }
            ]
        }
    )
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=1,
        _skip_env_load=True,
    )
    source_subtitle = SubtitleData(
        [SubtitleSegment("Hello world!", start_time=0, end_time=1000)]
    )

    with TranslationEngine(config, StubModelAdapter([response])) as engine:
        results = engine.translate_batch(source_subtitle, context_info="")

    assert results == [
        {
            "id": 1,
            "original": "Hello world!",
            "optimized": "Hello world!",
            "translation": "你好，世界！",
            "discarded": False,
        }
    ]


def test_closed_translation_engine_rejects_new_batches():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=1,
        _skip_env_load=True,
    )
    source_subtitle = SubtitleData(
        [SubtitleSegment("Hello world!", start_time=0, end_time=1000)]
    )
    engine = TranslationEngine(config, StubModelAdapter([]))
    engine.close()

    with pytest.raises(RuntimeError, match="closed"):
        engine.translate_batch(source_subtitle, context_info="")


def test_discarded_translation_result_does_not_trigger_fallback():
    response = json.dumps(
        {
            "subtitles": [
                {
                    "id": 1,
                    "optimized": "",
                    "translation": "",
                    "discarded": True,
                }
            ]
        }
    )
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=1,
        _skip_env_load=True,
    )
    source_subtitle = SubtitleData(
        [SubtitleSegment("Music.", start_time=0, end_time=1000)]
    )

    with TranslationEngine(config, StubModelAdapter([response])) as engine:
        results = engine.translate_batch(source_subtitle, context_info="")

    assert results == [
        {
            "id": 1,
            "original": "Music.",
            "optimized": "Music.",
            "translation": "",
            "discarded": True,
        }
    ]


def test_translation_run_shares_one_model_adapter_across_segmentation_and_translation(
    tmp_path,
):
    input_path = tmp_path / "lesson.srt"
    input_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello world!\n",
        encoding="utf-8",
    )
    translation_response = json.dumps(
        {
            "subtitles": [
                {
                    "id": 1,
                    "optimized": "Hello world!",
                    "translation": "你好，世界！",
                    "discarded": False,
                }
            ]
        }
    )
    adapter = StubModelAdapter(["Hello world!", translation_response])
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=1,
        min_batch_sentences=1,
        max_batch_sentences=1,
        max_batch_words=50,
        external_glossary_enabled=False,
        _skip_env_load=True,
    )
    translator = SubtitleTranslatorService(config=config, llm=adapter)

    output_path = translator.translate_srt(
        input_srt_path=input_path,
        target_lang="zh",
        output_dir=tmp_path / "output",
        skip_env_init=True,
    )

    assert output_path.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,000\n你好，世界！\n"
    )


def test_translation_batch_falls_back_to_a_retried_single_translation():
    failed_batch = json.dumps(
        {
            "subtitles": [
                {
                    "id": 1,
                    "optimized": "Hello world!",
                    "translation": "",
                    "discarded": False,
                }
            ]
        }
    )
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=1,
        _skip_env_load=True,
    )
    source_subtitle = SubtitleData(
        [SubtitleSegment("Hello world!", start_time=0, end_time=1000)]
    )
    adapter = StubModelAdapter(
        [failed_batch, failed_batch, RuntimeError("temporary failure"), "你好，世界！"]
    )

    with TranslationEngine(config, adapter) as engine:
        results = engine.translate_batch(source_subtitle, context_info="")

    assert results[0]["translation"] == "你好，世界！"


def test_batch_run_closes_owned_service_when_interrupted(monkeypatch, tmp_path):
    class FakeService:
        def __init__(self):
            self.closed = False

        def init_translation_env(self, **kwargs):
            pass

        def close(self):
            self.closed = True

    service = FakeService()
    monkeypatch.setattr(processor, "SubtitleTranslatorService", lambda: service)
    monkeypatch.setattr(
        processor,
        "process_single_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        processor.process_batch(
            files_to_process=[tmp_path / "lesson.srt"],
            target_lang="zh",
            output_dir=tmp_path,
            llm_model=None,
            split_model=None,
            translation_model=None,
            preserve_intermediate=False,
        )

    assert service.closed is True


def test_single_file_closes_owned_service_when_initialization_fails(
    monkeypatch,
    tmp_path,
):
    class FakeService:
        def __init__(self):
            self.closed = False

        def init_translation_env(self, *args, **kwargs):
            raise RuntimeError("invalid config")

        def close(self):
            self.closed = True

    service = FakeService()
    monkeypatch.setattr(processor, "SubtitleTranslatorService", lambda: service)

    with pytest.raises(RuntimeError, match="invalid config"):
        processor.process_single_file(
            input_file=tmp_path / "lesson.srt",
            target_lang="zh",
            output_dir=tmp_path,
            llm_model=None,
        )

    assert service.closed is True
