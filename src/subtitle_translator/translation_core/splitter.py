import difflib
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from ..logger import setup_logger
from .batch_utils import calculate_batch_sizes
from .config import SubtitleConfig
from .data import PreSplitSentence, SubtitleData, SubtitleSegment
from .llm_client import ModelAdapter
from .segmentation_rules import count_words, split_by_end_marks
from .split_by_llm import split_by_llm

logger = setup_logger("subtitle_merger")

# 常量定义
MAX_GAP = 1500  # 允许每个词语之间的最大时间间隔 ms
SIMILARITY_THRESHOLD = 0.5  # 相似度阈值
MAX_SHIFT = 30  # 滑动窗口的最大偏移量
MAX_UNMATCHED_SENTENCES = 5  # 允许的最大未匹配句子数量
SHORT_SEGMENT_TIME_GAP = 300  # 短分段合并的时间间隔阈值（毫秒）
SHORT_SEGMENT_MIN_WORDS = 5  # 短分段的最小单词数


class SubtitleSegmenter:
    """将 Source subtitle 转换为按 Translation batch 分组的 Sentence segment。"""

    def __init__(self, config: SubtitleConfig, llm: ModelAdapter):
        self.config = config
        self.llm = llm

    def segment(self, source_subtitle: SubtitleData) -> List[SubtitleData]:
        """完成预处理、分批、断句与 Timeline alignment。"""
        source_segments = preprocess_segments(list(source_subtitle.segments))
        prepared_subtitle = SubtitleData(source_segments)
        if not prepared_subtitle.is_word_timestamp():
            prepared_subtitle = prepared_subtitle.split_to_word_segments()

        word_segments = prepared_subtitle.segments
        pre_split_sentences = presplit_by_punctuation(word_segments)
        pre_split_sentences = hard_split_long_pre_split_sentences(
            pre_split_sentences,
            word_segments,
            self.config.max_batch_words,
        )
        batches = batch_by_sentence_count(
            pre_split_sentences,
            min_size=self.config.min_batch_sentences,
            max_size=self.config.max_batch_sentences,
            target_size=self.config.target_batch_sentences,
            max_words=self.config.max_batch_words,
        )

        if not batches:
            return []

        def segment_batch(batch_index: int, batch) -> SubtitleData:
            return SubtitleData(
                merge_segments_within_batch(
                    batch,
                    word_segments,
                    config=self.config,
                    llm=self.llm,
                    batch_index=batch_index + 1,
                )
            )

        with ThreadPoolExecutor(
            max_workers=min(len(batches), self.config.thread_num)
        ) as executor:
            return list(executor.map(segment_batch, range(len(batches)), batches))


def is_pure_punctuation(s: str) -> bool:
    """
    检查字符串是否仅由标点符号组成
    """
    return not re.search(r"\w", s, flags=re.UNICODE)


def preprocess_text(s: str) -> str:
    """
    通过规范化空格来标准化文本
    """
    return " ".join(s.split())


def presplit_by_punctuation(
    word_segments: List[SubtitleSegment],
) -> List[PreSplitSentence]:
    """基于标点预分句（移植自 youtube-subtitle）。"""
    if not word_segments:
        return []

    # 拼接所有单词为完整文本
    full_text = " ".join(seg.text for seg in word_segments)

    # 使用 split_by_end_marks 进行预分句
    sentences = split_by_end_marks(full_text)

    pre_split_sentences = []
    current_word_index = 0

    for sentence in sentences:
        sentence_words = sentence.strip().split()
        word_count = len(sentence_words)

        # 计算单词索引范围
        word_start_index = current_word_index
        word_end_index = current_word_index + word_count

        # 获取时间范围
        start_time = (
            word_segments[word_start_index].start_time
            if word_start_index < len(word_segments)
            else 0
        )
        end_time = (
            word_segments[min(word_end_index - 1, len(word_segments) - 1)].end_time
            if word_end_index > 0
            else 0
        )

        pre_split_sentences.append(
            PreSplitSentence(
                text=sentence,
                word_start_index=word_start_index,
                word_end_index=word_end_index,
                start_time=start_time,
                end_time=end_time,
            )
        )

        current_word_index = word_end_index

    return pre_split_sentences


def _build_pre_split_sentence_from_word_range(
    word_segments: List[SubtitleSegment],
    word_start_index: int,
    word_end_index: int,
) -> PreSplitSentence:
    """根据单词索引范围构造预分句。"""
    batch_word_segments = word_segments[word_start_index:word_end_index]
    text = " ".join(seg.text for seg in batch_word_segments)
    start_time = batch_word_segments[0].start_time if batch_word_segments else 0
    end_time = batch_word_segments[-1].end_time if batch_word_segments else 0
    return PreSplitSentence(
        text=text,
        word_start_index=word_start_index,
        word_end_index=word_end_index,
        start_time=start_time,
        end_time=end_time,
    )


def hard_split_long_pre_split_sentences(
    sentences: List[PreSplitSentence],
    word_segments: List[SubtitleSegment],
    max_words: int,
) -> List[PreSplitSentence]:
    """
    对超长预分句按词数硬切，避免单个预分句撑爆整个批次。
    """
    if max_words <= 0:
        return sentences

    split_sentences: List[PreSplitSentence] = []
    for sentence in sentences:
        sentence_word_count = sentence.word_end_index - sentence.word_start_index
        if sentence_word_count <= max_words:
            split_sentences.append(sentence)
            continue

        logger.warning(
            "⚠️ 预分句过长，按词数硬切: %s词 -> 每段≤%s词",
            sentence_word_count,
            max_words,
        )

        start_index = sentence.word_start_index
        while start_index < sentence.word_end_index:
            end_index = min(start_index + max_words, sentence.word_end_index)
            split_sentences.append(
                _build_pre_split_sentence_from_word_range(
                    word_segments,
                    start_index,
                    end_index,
                )
            )
            start_index = end_index

    return split_sentences


def batch_by_sentence_count(
    sentences: List[PreSplitSentence],
    min_size: int = 15,
    max_size: int = 25,
    target_size: Optional[int] = None,
    max_words: Optional[int] = None,
) -> List[List[PreSplitSentence]]:
    """按句子数分批；若配置了 max_words，则同时限制每批总词数。"""
    if not sentences:
        return []

    min_size = max(1, min(min_size, max_size))
    max_size = max(1, max_size)
    target_size = target_size or (min_size + max_size) // 2
    target_size = max(1, min(target_size, max_size))

    if max_words is not None and max_words > 0:
        batch_sizes = _balanced_batch_sizes_with_word_limit(
            sentences,
            min_size=min_size,
            max_size=max_size,
            target_size=target_size,
            max_words=max_words,
        )
    else:
        batch_sizes = calculate_batch_sizes(
            len(sentences), target_size, min_size, max_size
        )

    # 按计算出的批次大小分批
    batches = []
    start_index = 0
    for size in batch_sizes:
        batches.append(sentences[start_index : start_index + size])
        start_index += size

    return batches


def _balanced_batch_sizes_with_word_limit(
    sentences: List[PreSplitSentence],
    *,
    min_size: int,
    max_size: int,
    target_size: int,
    max_words: int,
) -> list[int]:
    """在句数和词数限制内，对连续预分句做尽量均衡的分区。"""
    sentence_count = len(sentences)
    word_counts = [
        max(0, sentence.word_end_index - sentence.word_start_index)
        for sentence in sentences
    ]
    prefix_words = [0]
    for word_count in word_counts:
        prefix_words.append(prefix_words[-1] + word_count)

    minimum_batch_count = max(
        math.ceil(sentence_count / target_size),
        math.ceil(prefix_words[-1] / max_words),
    )

    def solve(batch_count: int) -> Optional[list[int]]:
        ideal_sentences = sentence_count / batch_count
        ideal_words = prefix_words[-1] / batch_count
        states = {0: (0.0, [])}

        for completed_batches in range(batch_count):
            next_states = {}
            for start, (cost, sizes) in states.items():
                batches_left = batch_count - completed_batches - 1
                minimum_end = start + 1
                maximum_end = min(start + max_size, sentence_count)

                for end in range(minimum_end, maximum_end + 1):
                    remaining = sentence_count - end
                    if remaining < batches_left or remaining > batches_left * max_size:
                        continue

                    batch_words = prefix_words[end] - prefix_words[start]
                    batch_size = end - start
                    if batch_words > max_words and batch_size > 1:
                        break

                    sentence_delta = (batch_size - ideal_sentences) / target_size
                    word_scale = max(ideal_words, 1)
                    word_delta = (batch_words - ideal_words) / word_scale
                    small_batch_penalty = max(0, min_size - batch_size)
                    batch_cost = (
                        sentence_delta**2
                        + 0.35 * word_delta**2
                        + 0.05 * small_batch_penalty**2
                    )
                    candidate = (cost + batch_cost, [*sizes, batch_size])
                    previous = next_states.get(end)
                    if previous is None or candidate[0] < previous[0]:
                        next_states[end] = candidate
            states = next_states

        result = states.get(sentence_count)
        return result[1] if result else None

    for batch_count in range(minimum_batch_count, sentence_count + 1):
        batch_sizes = solve(batch_count)
        if batch_sizes is not None:
            return batch_sizes
    return [1] * sentence_count


def merge_segments_within_batch(
    pre_split_sentences: List[PreSplitSentence],
    word_segments: List[SubtitleSegment],
    config: SubtitleConfig,
    llm: ModelAdapter,
    model: Optional[str] = None,
    batch_index: Optional[int] = None,
) -> List[SubtitleSegment]:
    """在批次内进行 LLM 断句和时间戳对齐（移植自 youtube-subtitle）。"""
    if not pre_split_sentences:
        return []

    if model is None:
        model = config.llm_model

    # 提取批次对应的单词片段
    start_index = pre_split_sentences[0].word_start_index
    end_index = pre_split_sentences[-1].word_end_index
    batch_word_segments = word_segments[start_index:end_index]

    # 拼接为文本
    batch_text = " ".join(seg.text for seg in batch_word_segments)

    # 记录日志
    current_words = count_words(batch_text)
    batch_prefix = f"[批次{batch_index}]" if batch_index is not None else ""
    logger.info(
        f"📝 {batch_prefix} 处理 {current_words} 个单词，{len(pre_split_sentences)} 个预分句"
    )

    # LLM 断句
    llm_sentences = split_by_llm(
        batch_text,
        config=config,
        llm=llm,
        model=model,
        max_word_count_english=config.max_word_count_english,
        batch_index=batch_index,
    )
    logger.info(f"✂️ {batch_prefix} LLM 断句得到 {len(llm_sentences)} 个句子")

    # 时间戳对齐
    aligned_segments = merge_segments_based_on_sentences(
        batch_word_segments, llm_sentences
    )

    # 合并过短的分段
    merge_short_segment(aligned_segments, config.max_word_count_english)

    return aligned_segments


def merge_segments_based_on_sentences(
    segments: List[SubtitleSegment],
    sentences: List[str],
    max_unmatched: int = MAX_UNMATCHED_SENTENCES,
) -> List[SubtitleSegment]:
    """基于提供的句子列表合并字幕分段。"""
    asr_texts = [seg.text for seg in segments]
    asr_len = len(asr_texts)
    asr_index = 0
    unmatched_count = 0
    new_segments = []
    max_shift = MAX_SHIFT

    for sentence in sentences:
        sentence_proc = preprocess_text(sentence)
        sentence_key = re.sub(r"\W+", "", sentence_proc).casefold()
        word_count = count_words(sentence_proc)
        best_ratio = 0.0
        best_pos = None
        best_window_size = 0

        max_window_size = min(word_count * 2, asr_len - asr_index)
        min_window_size = max(1, word_count // 2)
        window_sizes = sorted(
            range(min_window_size, max_window_size + 1),
            key=lambda x: abs(x - word_count),
        )

        for window_size in window_sizes:
            max_start = min(asr_index + max_shift + 1, asr_len - window_size + 1)
            for start in range(asr_index, max_start):
                substr = "".join(asr_texts[start : start + window_size])
                substr_proc = re.sub(r"\W+", "", substr).casefold()
                ratio = difflib.SequenceMatcher(
                    None, sentence_key, substr_proc
                ).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pos = start
                    best_window_size = window_size
                if ratio == 1.0:
                    break
            if best_ratio == 1.0:
                break

        if best_ratio >= SIMILARITY_THRESHOLD and best_pos is not None:
            start_seg_index = best_pos
            end_seg_index = best_pos + best_window_size - 1

            # 模型漏掉的词段仍保留原文与时间轴。
            new_segments.extend(segments[asr_index:start_seg_index])
            segs_to_merge = segments[start_seg_index : end_seg_index + 1]
            seg_groups = merge_by_time_gaps(segs_to_merge, max_gap=MAX_GAP)

            source_key = re.sub(r"\W+", "", "".join(s.text for s in segs_to_merge)).casefold()
            for group in seg_groups:
                # 跨停顿或模型增删词时采用对应原文，避免复制整句或漏词。
                merged_text = (
                    sentence_proc if len(seg_groups) == 1 and source_key == sentence_key
                    else " ".join(s.text for s in group)
                )
                merged_start_time = group[0].start_time
                merged_end_time = group[-1].end_time
                merged_seg = SubtitleSegment(
                    merged_text, merged_start_time, merged_end_time
                )
                new_segments.append(merged_seg)

            max_shift = MAX_SHIFT
            asr_index = end_seg_index + 1
        else:
            logger.warning(f"无法匹配句子: {sentence}")
            unmatched_count += 1
            if unmatched_count > max_unmatched:
                logger.error(f"未匹配句子数量超过阈值 ({max_unmatched})，返回原始分段")
                return segments
            max_shift = 100

    if not new_segments:
        logger.warning("没有成功匹配任何句子，返回原始分段")
        return segments

    new_segments.extend(segments[asr_index:])
    return new_segments


def _should_merge_segments(current_seg, next_seg, max_word_count: int) -> bool:
    """判断是否应该合并两个分段。"""
    time_gap = abs(next_seg.start_time - current_seg.end_time)
    current_words = count_words(current_seg.text)
    next_words = count_words(next_seg.text)
    total_words = current_words + next_words

    # 判断条件：
    # 1. 时间间隔小于300ms
    # 2. 当前段落或下一段落词数小于5
    # 3. 合并后总词数不超过限制
    # 4. 当前段落不以句子结束标记结尾
    has_sentence_end = any(mark in current_seg.text for mark in [".", "?", "!"])

    return (
        time_gap < SHORT_SEGMENT_TIME_GAP
        and (
            current_words < SHORT_SEGMENT_MIN_WORDS
            or next_words <= SHORT_SEGMENT_MIN_WORDS
        )
        and total_words <= max_word_count
        and not has_sentence_end
    )


def merge_short_segment(
    segments: List[SubtitleSegment],
    max_word_count: int,
) -> None:
    """
    合并过短的分段
    """
    if not segments:
        return

    i = 0
    while i < len(segments) - 1:
        current_seg = segments[i]
        next_seg = segments[i + 1]

        if _should_merge_segments(current_seg, next_seg, max_word_count):
            # 执行合并操作
            logger.info(f"合并优化: {current_seg.text} --- {next_seg.text}")
            # 更新当前段落的文本和结束时间
            current_seg.text += " " + next_seg.text
            current_seg.end_time = next_seg.end_time

            # 从列表中移除下一个段落
            segments.pop(i + 1)
            # 不增加i，因为需要继续检查合并后的段落
        else:
            i += 1


def preprocess_segments(segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
    """预处理字幕分段。"""
    return [seg for seg in segments if not is_pure_punctuation(seg.text)]


def merge_by_time_gaps(
    segments: List[SubtitleSegment], max_gap: int = MAX_GAP
) -> List[List[SubtitleSegment]]:
    """按超过阈值的停顿将连续字幕分组。"""
    groups = []
    for segment in segments:
        if not groups or segment.start_time - groups[-1][-1].end_time > max_gap:
            groups.append([])
        groups[-1].append(segment)
    return groups
