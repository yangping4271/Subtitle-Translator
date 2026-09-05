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


@pytest.mark.parametrize("invalid_response", ["", "{}", "not JSON", "null"])
def test_empty_or_invalid_batch_response_retries_before_single_fallback(
    invalid_response,
):
    response = json.dumps({"1": {"translation": "你好，世界！"}})
    adapter = StubModelAdapter([invalid_response, response])
    batch = SubtitleData([SubtitleSegment("Hello world!", 0, 1000)])
    with TranslationEngine(
        SubtitleConfig(), adapter, DEFAULT_TRANSLATION_CONTEXT
    ) as engine:
        results = engine.translate_batch(batch, "")

    assert results == [
        {
            "id": 1,
            "original": "Hello world!",
            "optimized": "Hello world!",
            "translation": "你好，世界！",
            "discarded": False,
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

    adapter.create_chat_completion = Mock(wraps=adapter.create_chat_completion)
    with TranslationEngine(config, adapter, DEFAULT_TRANSLATION_CONTEXT) as engine:
        results = engine.translate_batch(source_subtitle, context_info="Course introduction")

    assert results[0]["translation"] == "你好，世界！"
    assert adapter.create_chat_completion.call_count == 4
    messages = adapter.create_chat_completion.call_args.kwargs["messages"]
    assert "<reference>Course introduction</reference>" in messages[-1]["content"]
    assert "<subtitles>Hello world!</subtitles>" in messages[-1]["content"]


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


@pytest.mark.parametrize("single_succeeds", [True, False])
def test_partial_retry_preserves_order_discard_and_unrecovered_subtitles(
    single_succeeds,
):
    initial = json.dumps(
        {
            "subtitles": [
                {"id": 4, "translation": None},
                {"id": 2, "discarded": True},
                {"id": 1, "translation": "第一句"},
            ]
        }
    )
    retry_response = json.dumps(
        {
            "subtitles": [
                {"id": 3, "translation": "第三句"},
                {"id": 4, "translation": ""},
            ]
        }
    )
    single_responses = (
        ["第四句"]
        if single_succeeds
        else [RuntimeError("unavailable"), RuntimeError("unavailable")]
    )
    adapter = StubModelAdapter([initial, retry_response, *single_responses])
    source = SubtitleData(
        [
            SubtitleSegment(text, i * 1000, (i + 1) * 1000)
            for i, text in enumerate(
                ["First sentence.", "Music.", "Third sentence.", "Fourth sentence."]
            )
        ]
    )
    config = SubtitleConfig(openai_base_url="https://api.openai.com/v1", thread_num=1)
    with TranslationEngine(config, adapter, DEFAULT_TRANSLATION_CONTEXT) as engine:
        results = engine.translate_batch(source, context_info="")
    assert [r["id"] for r in results] == [1, 2, 3, 4]
    assert [r["translation"] for r in results] == [
        "第一句",
        "",
        "第三句",
        "第四句" if single_succeeds else "",
    ]
    assert [r["discarded"] for r in results] == [False, True, False, False]
    assert [r["optimized"] for r in results] == [s.text for s in source]
    assert [r["original"] for r in results] == [s.text for s in source]


def test_single_file_initializes_once_and_closes_owned_service(monkeypatch, tmp_path):
    service = Mock()
    service.translate_srt.return_value = SimpleNamespace(
        bilingual_ass=tmp_path / "lesson.ass", intermediates_preserved=False
    )
    monkeypatch.setattr(processor, "SubtitleTranslatorService", lambda: service)
    processor.process_single_file(tmp_path / "lesson.srt", "zh", tmp_path, "test-model")
    service.init_translation_env.assert_called_once_with("test-model", show_config=True)
    assert service.translate_srt.call_args.kwargs["skip_env_init"] is True
    service.close.assert_called_once()


def test_concurrent_translation_keeps_batch_order_and_global_ids(monkeypatch):
    from threading import Event
    from subtitle_translator import service as service_module

    batches = [
        SubtitleData([SubtitleSegment(str(i), i * 1000, i * 1000 + 900)])
        for i in range(5)
    ]
    second_finished = Event()

    def translate(self, batch, context_info, batch_num, total_batches):
        assert total_batches == 5
        if batch_num == 1:
            assert second_finished.wait(2)
        if batch_num == 2:
            second_finished.set()
        return [
            {
                "id": 1,
                "original": batch.segments[0].text,
                "optimized": batch.segments[0].text,
                "translation": str(batch_num),
                "discarded": False,
            }
        ]

    monkeypatch.setattr(
        service_module.SubtitleSegmenter, "segment", lambda *args: batches
    )
    monkeypatch.setattr(TranslationEngine, "translate_batch", translate)
    service = SubtitleTranslatorService(
        config=SubtitleConfig(thread_num=2), llm=StubModelAdapter([])
    )
    sentences, results, _ = service._translate_segmented_batches(
        SubtitleData([]), "", DEFAULT_TRANSLATION_CONTEXT
    )
    assert [s.text for s in sentences] == [str(i) for i in range(5)]
    assert [r["id"] for r in results] == list(range(1, 6))
    assert [r["original"] for r in results] == [s.text for s in sentences]
    assert [r["translation"] for r in results] == [str(i) for i in range(1, 6)]
