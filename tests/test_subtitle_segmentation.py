from types import SimpleNamespace

import pytest

from subtitle_translator.exceptions import SmartSplitError
from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core.data import SubtitleData, SubtitleSegment
from subtitle_translator.translation_core.split_by_llm import split_by_llm
from subtitle_translator.translation_core.splitter import SubtitleSegmenter


class StubModelAdapter:
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)

    def create_chat_completion(self, **kwargs):
        content = next(self._responses)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class EchoModelAdapter:
    def create_chat_completion(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        content = prompt.rsplit("\n", 1)[-1]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FailingModelAdapter:
    def create_chat_completion(self, **kwargs):
        raise RuntimeError("model unavailable")


def test_source_subtitle_becomes_time_aligned_sentence_segments():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=1,
        min_batch_sentences=1,
        max_batch_sentences=1,
        max_batch_words=50,
        _skip_env_load=True,
    )
    source_subtitle = SubtitleData(
        [
            SubtitleSegment("Hello", start_time=0, end_time=500),
            SubtitleSegment("world!", start_time=500, end_time=1000),
        ]
    )
    segmenter = SubtitleSegmenter(config, StubModelAdapter(["Hello world!"]))

    batches = segmenter.segment(source_subtitle)

    assert len(batches) == 1
    assert [
        (segment.text, segment.start_time, segment.end_time)
        for segment in batches[0].segments
    ] == [("Hello world!", 0, 1000)]


def test_mixed_language_segments_share_one_counting_policy():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=1,
        min_batch_sentences=1,
        max_batch_sentences=1,
        max_batch_words=50,
        max_word_count_english=1,
        _skip_env_load=True,
    )
    source_subtitle = SubtitleData(
        [
            SubtitleSegment("Hello", start_time=0, end_time=100),
            SubtitleSegment("世", start_time=100, end_time=200),
            SubtitleSegment("界", start_time=200, end_time=300),
        ]
    )
    segmenter = SubtitleSegmenter(config, StubModelAdapter(["Hello 世界"]))

    batches = segmenter.segment(source_subtitle)

    assert [segment.text for segment in batches[0].segments] == ["Hello", "世界"]


def test_sentence_batches_keep_source_timeline_order():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=2,
        min_batch_sentences=1,
        max_batch_sentences=1,
        max_batch_words=50,
        _skip_env_load=True,
    )
    source_subtitle = SubtitleData(
        [
            SubtitleSegment("First", start_time=0, end_time=100),
            SubtitleSegment("short", start_time=100, end_time=200),
            SubtitleSegment("sentence.", start_time=200, end_time=300),
            SubtitleSegment("Second", start_time=400, end_time=500),
            SubtitleSegment("short", start_time=500, end_time=600),
            SubtitleSegment("sentence.", start_time=600, end_time=700),
        ]
    )

    batches = SubtitleSegmenter(config, EchoModelAdapter()).segment(source_subtitle)

    assert [batch.segments[0].text for batch in batches] == [
        "First short sentence.",
        "Second short sentence.",
    ]
    assert [batch.segments[0].start_time for batch in batches] == [0, 400]


def test_segmentation_raises_smart_split_error_after_model_retries_fail():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=1,
        min_batch_sentences=1,
        max_batch_sentences=1,
        max_batch_words=50,
        _skip_env_load=True,
    )
    source_subtitle = SubtitleData(
        [
            SubtitleSegment("Hello", start_time=0, end_time=500),
            SubtitleSegment("world!", start_time=500, end_time=1000),
        ]
    )

    with pytest.raises(SmartSplitError, match="model unavailable"):
        SubtitleSegmenter(config, FailingModelAdapter()).segment(source_subtitle)


def test_llm_postprocessing_keeps_original_explicit_end_mark_policy():
    text = "Alpha beta gamma.Delta epsilon zeta"
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        _skip_env_load=True,
    )

    sentences = split_by_llm(
        text,
        config=config,
        llm=StubModelAdapter([text]),
        max_word_count_english=20,
    )

    assert sentences == [text]
