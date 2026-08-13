import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from subtitle_translator.exceptions import TranslationError
from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core.data import SubtitleData, SubtitleSegment
from subtitle_translator.translation_core.llm_client import LLMClient, RequestMetric
from subtitle_translator.translation_core.translation_execution import TranslationEngine
from subtitle_translator.translation_core.translation_context import TranslationContext
from subtitle_translator import processor
from subtitle_translator.service import SubtitleTranslatorService


DEFAULT_TRANSLATION_CONTEXT = TranslationContext(target_language="简体中文")


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
    )
    source_subtitle = SubtitleData(
        [SubtitleSegment("Hello world!", start_time=0, end_time=1000)]
    )

    with TranslationEngine(
        config,
        StubModelAdapter([response]),
        DEFAULT_TRANSLATION_CONTEXT,
    ) as engine:
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
    )
    source_subtitle = SubtitleData(
        [SubtitleSegment("Hello world!", start_time=0, end_time=1000)]
    )
    engine = TranslationEngine(
        config,
        StubModelAdapter([]),
        DEFAULT_TRANSLATION_CONTEXT,
    )
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
    )
    source_subtitle = SubtitleData(
        [SubtitleSegment("Music.", start_time=0, end_time=1000)]
    )

    with TranslationEngine(
        config,
        StubModelAdapter([response]),
        DEFAULT_TRANSLATION_CONTEXT,
    ) as engine:
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
    )
    translator = SubtitleTranslatorService(config=config, llm=adapter)
    original_config = dict(vars(config))

    outputs = translator.translate_srt(
        input_srt_path=input_path,
        target_lang="zh",
        output_dir=tmp_path / "output",
        skip_env_init=True,
    )

    assert outputs.target_srt.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,000\n你好，世界！\n"
    )
    assert outputs.source_srt.exists()
    assert outputs.bilingual_ass.exists()
    assert outputs.intermediates_preserved is True
    assert vars(config) == original_config


def test_batch_run_uses_the_ass_path_returned_by_single_file(
    monkeypatch,
    tmp_path,
):
    class FakeService:
        def init_translation_env(self, **kwargs):
            pass

        def close(self):
            pass

    returned_ass = tmp_path / "nonstandard-name.ass"
    returned_ass.write_text("ass", encoding="utf-8")
    output = SimpleNamespace(bilingual_ass=returned_ass)
    shown_results = {}

    monkeypatch.setattr(processor, "SubtitleTranslatorService", FakeService)
    monkeypatch.setattr(
        processor, "process_single_file", lambda *args, **kwargs: output
    )
    monkeypatch.setattr(
        processor,
        "show_results",
        lambda count, files, output_dir, batch_mode: shown_results.update(
            count=count,
            files=files,
            output_dir=output_dir,
            batch_mode=batch_mode,
        ),
    )

    processor.process_batch(
        files_to_process=[tmp_path / "lesson.srt"],
        target_lang="zh",
        output_dir=tmp_path,
        llm_model=None,
        split_model=None,
        translation_model=None,
        preserve_intermediate=False,
    )

    assert shown_results["files"] == [returned_ass]


def test_batch_progress_shows_one_translation_line_with_filename(
    monkeypatch,
    tmp_path,
    capsys,
):
    class FakeService:
        def init_translation_env(self, **kwargs):
            pass

        def close(self):
            pass

    output = SimpleNamespace(bilingual_ass=tmp_path / "lesson.ass")
    monkeypatch.setattr(processor, "SubtitleTranslatorService", FakeService)
    monkeypatch.setattr(
        processor,
        "process_single_file",
        lambda *args, **kwargs: output,
    )
    monkeypatch.setattr(processor, "show_results", lambda *args, **kwargs: None)

    processor.process_batch(
        files_to_process=[tmp_path / "lesson.srt", tmp_path / "next.srt"],
        target_lang="zh",
        output_dir=tmp_path,
        llm_model=None,
        split_model=None,
        translation_model=None,
        preserve_intermediate=False,
    )

    console_output = capsys.readouterr().out
    assert "🎯 开始翻译第 1/2 个文件: lesson.srt" in console_output
    assert console_output.count("lesson.srt") == 1


def test_single_file_processing_does_not_repeat_console_progress(
    tmp_path,
    capsys,
):
    class FakeService:
        def translate_srt(self, **kwargs):
            return SimpleNamespace(
                bilingual_ass=tmp_path / "lesson.ass",
                target_srt=tmp_path / "lesson.zh.srt",
                source_srt=tmp_path / "lesson.en.srt",
                intermediates_preserved=False,
            )

    processor.process_single_file(
        input_file=tmp_path / "lesson.srt",
        target_lang="zh",
        output_dir=tmp_path,
        llm_model=None,
        translator_service=FakeService(),
    )

    output = capsys.readouterr().out
    assert ">>> 检测到 SRT 文件" not in output
    assert ">>> 开始翻译: lesson.srt" not in output
    assert ">>> 开始翻译..." not in output


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
    )
    source_subtitle = SubtitleData(
        [SubtitleSegment("Hello world!", start_time=0, end_time=1000)]
    )
    adapter = StubModelAdapter(
        [failed_batch, failed_batch, RuntimeError("temporary failure"), "你好，世界！"]
    )

    with TranslationEngine(config, adapter, DEFAULT_TRANSLATION_CONTEXT) as engine:
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


def test_failed_translation_still_shows_api_metrics(tmp_path, capsys, monkeypatch):
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    service = SubtitleTranslatorService(config=config, llm=client)

    def fail_after_metric(*args, **kwargs):
        client._record_metric(
            RequestMetric(
                started_at=0.0,
                ended_at=12.0,
                latency=12.0,
                success=False,
                completion_tokens=None,
                finish_reason=None,
                content_chars=0,
                error_type="APITimeoutError",
            )
        )
        raise TranslationError("translation failed")

    monkeypatch.setattr(service, "_load_translation_context", fail_after_metric)

    with pytest.raises(TranslationError, match="translation failed"):
        service.translate_srt(
            input_srt_path=tmp_path / "lesson.srt",
            target_lang="zh",
            output_dir=tmp_path,
            skip_env_init=True,
        )

    output = capsys.readouterr().out
    assert "API 性能统计" in output
    assert "请求: 1 次 (成功 0 / 失败 1)" in output


def test_metrics_display_failure_does_not_mask_translation_error(
    tmp_path,
    monkeypatch,
):
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
    )
    client = LLMClient(config)
    service = SubtitleTranslatorService(config=config, llm=client)
    monkeypatch.setattr(
        service,
        "_load_translation_context",
        Mock(side_effect=TranslationError("original boom")),
    )
    monkeypatch.setattr(
        client,
        "metrics_summary",
        Mock(side_effect=RuntimeError("metrics boom")),
    )

    with pytest.raises(TranslationError, match="original boom"):
        service.translate_srt(
            input_srt_path=tmp_path / "lesson.srt",
            target_lang="zh",
            output_dir=tmp_path,
            skip_env_init=True,
        )


def test_metrics_display_failure_does_not_fail_successful_translation(
    tmp_path,
    monkeypatch,
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
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
        thread_num=1,
        min_batch_sentences=1,
        max_batch_sentences=1,
        max_batch_words=50,
        external_glossary_enabled=False,
    )
    client = LLMClient(config)
    stub = StubModelAdapter(["Hello world!", translation_response])
    client.create_chat_completion = stub.create_chat_completion
    client._record_metric(
        RequestMetric(
            started_at=0.0,
            ended_at=1.0,
            latency=1.0,
            success=True,
            completion_tokens=8,
            finish_reason="stop",
            content_chars=12,
            error_type=None,
        )
    )
    service = SubtitleTranslatorService(config=config, llm=client)
    monkeypatch.setattr(
        client,
        "metrics_summary",
        Mock(side_effect=RuntimeError("metrics boom")),
    )

    outputs = service.translate_srt(
        input_srt_path=input_path,
        target_lang="zh",
        output_dir=tmp_path / "output",
        skip_env_init=True,
    )

    assert outputs.bilingual_ass.exists()


def test_successful_translation_shows_api_metrics_once(
    tmp_path,
    capsys,
    monkeypatch,
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
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
        thread_num=1,
        min_batch_sentences=1,
        max_batch_sentences=1,
        max_batch_words=50,
        external_glossary_enabled=False,
    )
    client = LLMClient(config)
    stub = StubModelAdapter(["Hello world!", translation_response])
    client.create_chat_completion = stub.create_chat_completion
    service = SubtitleTranslatorService(config=config, llm=client)

    def record_then_translate(*args, **kwargs):
        client._record_metric(
            RequestMetric(
                started_at=0.0,
                ended_at=2.0,
                latency=2.0,
                success=True,
                completion_tokens=8,
                finish_reason="stop",
                content_chars=12,
                error_type=None,
            )
        )
        return original_translate(*args, **kwargs)

    original_translate = service._translate_segmented_batches
    monkeypatch.setattr(service, "_translate_segmented_batches", record_then_translate)

    service.translate_srt(
        input_srt_path=input_path,
        target_lang="zh",
        output_dir=tmp_path / "output",
        skip_env_init=True,
    )

    output = capsys.readouterr().out
    assert output.count("API 性能统计") == 1


def test_process_batch_does_not_repeat_start_filename_in_app_log(
    monkeypatch,
    tmp_path,
):
    class FakeService:
        def init_translation_env(self, **kwargs):
            pass

        def close(self):
            pass

        def translate_srt(self, **kwargs):
            return SimpleNamespace(
                bilingual_ass=tmp_path / "lesson.ass",
                target_srt=tmp_path / "lesson.zh.srt",
                source_srt=tmp_path / "lesson.en.srt",
                intermediates_preserved=False,
            )

    (tmp_path / "lesson.ass").write_text("ass", encoding="utf-8")
    monkeypatch.setattr(processor, "SubtitleTranslatorService", FakeService)
    monkeypatch.setattr(processor, "show_results", lambda *args, **kwargs: None)
    log_messages = []
    original_info = processor.logger.info

    def capture_info(message, *args, **kwargs):
        log_messages.append(str(message) % args if args else str(message))
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(processor.logger, "info", capture_info)

    processor.process_batch(
        files_to_process=[tmp_path / "lesson.srt"],
        target_lang="zh",
        output_dir=tmp_path,
        llm_model=None,
        split_model=None,
        translation_model=None,
        preserve_intermediate=False,
    )

    start_translation_logs = [
        message
        for message in log_messages
        if "开始翻译" in message and "lesson.srt" in message
    ]
    start_progress_logs = [
        message
        for message in log_messages
        if "处理文件" in message and "lesson.srt" in message
    ]
    assert start_translation_logs == []
    assert len(start_progress_logs) == 1
    assert not any(">>> 开始翻译" in message for message in log_messages)
