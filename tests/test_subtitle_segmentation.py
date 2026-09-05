from types import SimpleNamespace

import pytest

from subtitle_translator.exceptions import SmartSplitError
from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core.data import (
    PreSplitSentence,
    SubtitleData,
    SubtitleSegment,
)
from subtitle_translator.translation_core.split_by_llm import split_by_llm
from subtitle_translator.translation_core.splitter import (
    SubtitleSegmenter,
    batch_by_sentence_count,
)


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


def _pre_split_sentences(word_counts: list[int]) -> list[PreSplitSentence]:
    sentences = []
    start = 0
    for index, word_count in enumerate(word_counts):
        end = start + word_count
        sentences.append(
            PreSplitSentence(
                text=f"sentence {index}",
                word_start_index=start,
                word_end_index=end,
                start_time=index * 100,
                end_time=(index + 1) * 100,
            )
        )
        start = end
    return sentences


def test_word_limited_batches_balance_small_tail():
    sentences = _pre_split_sentences([12] * 27)

    batches = batch_by_sentence_count(
        sentences,
        min_size=15,
        max_size=25,
        target_size=20,
        max_words=500,
    )

    assert sorted(len(batch) for batch in batches) == [13, 14]


def test_balanced_batches_respect_word_and_sentence_limits():
    sentences = _pre_split_sentences([20, 20, 20, 80, 20, 20, 20])

    batches = batch_by_sentence_count(
        sentences,
        min_size=2,
        max_size=4,
        target_size=3,
        max_words=100,
    )

    assert [sentence for batch in batches for sentence in batch] == sentences
    assert all(len(batch) <= 4 for batch in batches)
    assert all(
        sum(s.word_end_index - s.word_start_index for s in batch) <= 100
        for batch in batches
    )


def test_source_subtitle_becomes_time_aligned_sentence_segments():
    config = SubtitleConfig(
        openai_base_url="https://api.openai.com/v1",
        thread_num=1,
        min_batch_sentences=1,
        max_batch_sentences=1,
        max_batch_words=50,
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
    )

    sentences = split_by_llm(
        text,
        config=config,
        llm=StubModelAdapter([text]),
        max_word_count_english=20,
    )

    assert sentences == [text]


@pytest.mark.parametrize("word_count", [14, 16, 21, 28, 29, 50])
def test_long_sentence_postprocessing_preserves_every_word(word_count):
    text = " ".join(f"word{i}" for i in range(word_count))
    sentences = split_by_llm(text, SubtitleConfig(), StubModelAdapter([text]))

    assert " ".join(sentences) == text
    if word_count <= 21:  # 警告阈值内没有语义边界时保留原句。
        assert sentences == [text]
    else:
        assert len(sentences) > 1
        assert all(len(sentence.split()) <= 21 for sentence in sentences)


def test_long_sentence_prefers_conjunction_boundary():
    text = "one two three four five six seven and eight nine ten eleven twelve"
    sentences = split_by_llm(
        text, SubtitleConfig(), StubModelAdapter([text]), max_word_count_english=7
    )

    assert sentences == [
        "one two three four five six seven",
        "and eight nine ten eleven twelve",
    ]


def test_explicit_end_marks_preserve_numbers_and_attach_short_tail():
    text = "Version 3. one two three. Four five six! End"
    sentences = split_by_llm(
        text, SubtitleConfig(), StubModelAdapter([text]), max_word_count_english=20
    )

    assert sentences == ["Version 3. one two three.", "Four five six! End"]
