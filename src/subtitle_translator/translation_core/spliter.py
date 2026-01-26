import difflib
import re
from typing import List, Optional

from ..logger import setup_logger
from .batch_utils import calculate_batch_sizes
from .config import get_default_config
from .data import PreSplitSentence, SubtitleSegment
from .split_by_llm import split_by_llm

logger = setup_logger("subtitle_merger")

# 常量定义
MAX_GAP = 1500  # 允许每个词语之间的最大时间间隔 ms
MIN_SENTENCE_WORDS = 3  # 分句的最小单词数
MIN_LAST_SEGMENT_WORDS = 2  # 最后一段的最小单词数
SIMILARITY_THRESHOLD = 0.5  # 相似度阈值
MAX_SHIFT = 30  # 滑动窗口的最大偏移量
MAX_UNMATCHED_SENTENCES = 5  # 允许的最大未匹配句子数量
SHORT_SEGMENT_TIME_GAP = 300  # 短分段合并的时间间隔阈值（毫秒）
SHORT_SEGMENT_MIN_WORDS = 5  # 短分段的最小单词数

def is_pure_punctuation(s: str) -> bool:
    """
    检查字符串是否仅由标点符号组成
    """
    return not re.search(r'\w', s, flags=re.UNICODE)


def count_words(text: str) -> int:
    """
    统计多语言文本中的字符/单词数
    """
    # 定义各种语言的Unicode范围
    patterns = [
        r'[\u4e00-\u9fff]',           # 中日韩统一表意文字
        r'[\u3040-\u309f]',           # 平假名
        r'[\u30a0-\u30ff]',           # 片假名
        r'[\uac00-\ud7af]',           # 韩文音节
        r'[\u0e00-\u0e7f]',           # 泰文
        r'[\u0600-\u06ff]',           # 阿拉伯文
        r'[\u0400-\u04ff]',           # 西里尔字母（俄文等）
        r'[\u0590-\u05ff]',           # 希伯来文
        r'[\u1e00-\u1eff]',           # 越南文
        r'[\u3130-\u318f]',           # 韩文兼容字母
    ]
    
    # 统计所有非英文字符
    non_english_chars = 0
    remaining_text = text
    
    for pattern in patterns:
        # 计算当前语言的字符数
        chars = len(re.findall(pattern, remaining_text))
        non_english_chars += chars
        # 从文本中移除已计数的字符
        remaining_text = re.sub(pattern, ' ', remaining_text)
    
    # 计算英文单词数（处理剩余的文本）
    english_words = len(remaining_text.strip().split())
    
    return non_english_chars + english_words


def preprocess_text(s: str) -> str:
    """
    通过规范化空格来标准化文本
    """
    return ' '.join(s.split())


def split_by_end_marks(sentence: str) -> List[str]:
    """
    按句子结束标记拆分句子（移植自 youtube-subtitle）

    规则：
    - 按 '. ' '! ' '? ' '.' '!' '?' 分句
    - 跳过小数点（检测前一个字符是否为数字）
    - 每段至少 MIN_SENTENCE_WORDS 个单词才分割
    - 最后一段少于 MIN_LAST_SEGMENT_WORDS 个单词时，合并到前一段

    Args:
        sentence: 输入文本

    Returns:
        拆分后的句子列表
    """
    end_marks = ['. ', '! ', '? ', '.', '!', '?']
    positions = []

    for mark in end_marks:
        start = 0
        while True:
            pos = sentence.find(mark, start)
            if pos == -1:
                break

            if _is_decimal_point(sentence, pos, mark):
                start = pos + 1
                continue

            positions.append(pos + len(mark))
            start = pos + 1

    if not positions:
        return [sentence]

    unique_positions = sorted(set(positions))
    segments = []
    start = 0

    for pos in unique_positions:
        segment = sentence[start:pos].strip()
        if segment and count_words(segment) >= MIN_SENTENCE_WORDS:
            segments.append(segment)
            start = pos

    last_segment = sentence[start:].strip()
    if last_segment:
        if segments and count_words(last_segment) < MIN_LAST_SEGMENT_WORDS:
            segments[-1] = segments[-1] + ' ' + last_segment
        else:
            segments.append(last_segment)

    return segments if len(segments) > 1 else [sentence]


def _is_decimal_point(sentence: str, pos: int, mark: str) -> bool:
    """检查是否为小数点"""
    return (mark == '. ' or mark == '.') and pos > 0 and sentence[pos - 1].isdigit()


def presplit_by_punctuation(word_segments: List[SubtitleSegment]) -> List[PreSplitSentence]:
    """
    基于标点预分句（移植自 youtube-subtitle）

    Args:
        word_segments: 单词级字幕段列表

    Returns:
        PreSplitSentence 列表，包含句子文本和对应的单词索引范围
    """
    if not word_segments:
        return []

    # 拼接所有单词为完整文本
    full_text = ' '.join(seg.text for seg in word_segments)

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
        start_time = word_segments[word_start_index].start_time if word_start_index < len(word_segments) else 0
        end_time = word_segments[min(word_end_index - 1, len(word_segments) - 1)].end_time if word_end_index > 0 else 0

        pre_split_sentences.append(PreSplitSentence(
            text=sentence,
            word_start_index=word_start_index,
            word_end_index=word_end_index,
            start_time=start_time,
            end_time=end_time
        ))

        current_word_index = word_end_index

    return pre_split_sentences


def batch_by_sentence_count(
    sentences: List[PreSplitSentence],
    min_size: int = 15,
    max_size: int = 25
) -> List[List[PreSplitSentence]]:
    """
    按句子数分批（移植自 youtube-subtitle，移除首批特殊处理）

    Args:
        sentences: 预分句列表
        min_size: 最小批次大小
        max_size: 最大批次大小

    Returns:
        批次列表，每个批次是 PreSplitSentence 列表
    """
    if not sentences:
        return []

    # 计算批次大小（不使用首批特殊处理）
    target_size = (min_size + max_size) // 2
    batch_sizes = calculate_batch_sizes(len(sentences), target_size, min_size, max_size)

    # 按计算出的批次大小分批
    batches = []
    start_index = 0
    for size in batch_sizes:
        batches.append(sentences[start_index:start_index + size])
        start_index += size

    return batches


def merge_segments_within_batch(
    pre_split_sentences: List[PreSplitSentence],
    word_segments: List[SubtitleSegment],
    model: Optional[str] = None,
    batch_index: Optional[int] = None
) -> List[SubtitleSegment]:
    """
    在批次内进行 LLM 断句和时间戳对齐（移植自 youtube-subtitle）

    Args:
        pre_split_sentences: 批次内的预分句列表
        word_segments: 完整的单词级字幕段（用于时间戳对齐）
        model: LLM 模型名称
        batch_index: 批次索引（用于日志）

    Returns:
        处理后的字幕段列表
    """
    if not pre_split_sentences:
        return []

    config = get_default_config()
    if model is None:
        model = config.split_model

    # 提取批次对应的单词片段
    start_index = pre_split_sentences[0].word_start_index
    end_index = pre_split_sentences[-1].word_end_index
    batch_word_segments = word_segments[start_index:end_index]

    # 拼接为文本
    batch_text = ' '.join(seg.text for seg in batch_word_segments)

    # 记录日志
    current_words = count_words(batch_text)
    batch_prefix = f"[批次{batch_index}]" if batch_index is not None else ""
    logger.info(f"📝 {batch_prefix} 处理 {current_words} 个单词，{len(pre_split_sentences)} 个预分句")

    # LLM 断句
    llm_sentences = split_by_llm(
        batch_text,
        model=model,
        max_word_count_english=config.max_word_count_english,
        batch_index=batch_index
    )
    logger.info(f"✂️ {batch_prefix} LLM 断句得到 {len(llm_sentences)} 个句子")

    # 时间戳对齐
    aligned_segments = merge_segments_based_on_sentences(batch_word_segments, llm_sentences)

    # 合并过短的分段
    merge_short_segment(aligned_segments)

    return aligned_segments



def merge_segments_based_on_sentences(segments: List[SubtitleSegment], sentences: List[str], max_unmatched: int = MAX_UNMATCHED_SENTENCES) -> List[SubtitleSegment]:
    """
    基于提供的句子列表合并字幕分段
    
    Args:
        segments: 字幕段列表
        sentences: 句子列表
        max_unmatched: 允许的最大未匹配句子数量，超过此数量将抛出异常
        
    Returns:
        合并后的 SubtitleSegment 列表
        
    Raises:
        SubtitleProcessError: 当未匹配句子数量超过阈值时抛出
    """
    asr_texts = [seg.text for seg in segments]
    asr_len = len(asr_texts)
    asr_index = 0
    unmatched_count = 0
    new_segments = []
    max_shift = MAX_SHIFT

    for sentence in sentences:
        sentence_proc = preprocess_text(sentence)
        word_count = count_words(sentence_proc)
        best_ratio = 0.0
        best_pos = None
        best_window_size = 0

        max_window_size = min(word_count * 2, asr_len - asr_index)
        min_window_size = max(1, word_count // 2)
        window_sizes = sorted(range(min_window_size, max_window_size + 1), key=lambda x: abs(x - word_count))

        for window_size in window_sizes:
            max_start = min(asr_index + max_shift + 1, asr_len - window_size + 1)
            for start in range(asr_index, max_start):
                substr = ''.join(asr_texts[start:start + window_size])
                substr_proc = preprocess_text(substr)
                ratio = difflib.SequenceMatcher(None, sentence_proc, substr_proc).ratio()

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

            segs_to_merge = segments[start_seg_index:end_seg_index + 1]
            seg_groups = merge_by_time_gaps(segs_to_merge, max_gap=MAX_GAP)

            for group in seg_groups:
                merged_text = sentence_proc
                merged_start_time = group[0].start_time
                merged_end_time = group[-1].end_time
                merged_seg = SubtitleSegment(merged_text, merged_start_time, merged_end_time)
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
            asr_index = min(asr_index + 1, asr_len - 1)

    if not new_segments:
        logger.warning("没有成功匹配任何句子，返回原始分段")
        return segments

    return new_segments

def merge_short_segment(segments: List[SubtitleSegment]) -> None:
    """
    合并过短的分段
    """
    if not segments:  # 添加空列表检查
        return
        
    i = 0  # 从头开始遍历
    while i < len(segments) - 1:  # 修改遍历方式
        current_seg = segments[i]
        next_seg = segments[i + 1]
        
        # 判断是否需要合并:
        # 1. 时间间隔小于300ms
        # 2. 当前段落或下一段落词数小于5
        # 3. 合并后总词数不超过限制
        time_gap = abs(next_seg.start_time - current_seg.end_time)
        current_words = count_words(current_seg.text)
        next_words = count_words(next_seg.text)
        total_words = current_words + next_words
        config = get_default_config()
        max_word_count = config.max_word_count_english

        if time_gap < 300 and (current_words < 5 or next_words <= 5) \
            and total_words <= max_word_count \
            and ("." not in current_seg.text and "?" not in current_seg.text and "!" not in current_seg.text):
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
    """
    预处理字幕分段:
    1. 移除纯标点符号的分段
    2. 保留原始大小写格式
    
    Args:
        segments: 字幕分段列表
    Returns:
        处理后的分段列表
    """
    new_segments = []
    for seg in segments:
        if not is_pure_punctuation(seg.text):
            # 保留原始格式，不转换为小写
            new_segments.append(seg)
    return new_segments


def merge_by_time_gaps(segments: List[SubtitleSegment], max_gap: int = MAX_GAP, check_large_gaps: bool = False) -> List[List[SubtitleSegment]]:
    """
    根据时间间隔合并分段
    """
    if not segments:
        return []

    result = []
    current_group = [segments[0]]
    recent_gaps = []  # 存储最近的时间间隔
    WINDOW_SIZE = 5   # 检查最近5个间隔

    for i in range(1, len(segments)):
        time_gap = segments[i].start_time - segments[i-1].end_time

        if check_large_gaps:
            recent_gaps.append(time_gap)
            if len(recent_gaps) > WINDOW_SIZE:
                recent_gaps.pop(0)
            if len(recent_gaps) == WINDOW_SIZE:
                avg_gap = sum(recent_gaps) / len(recent_gaps)
                # 如果当前间隔大于平均值的3倍
                if time_gap > avg_gap*3 and len(current_group) > 5:
                    result.append(current_group)
                    current_group = []
                    recent_gaps = []  # 重置间隔记录

        if time_gap > max_gap:
            result.append(current_group)
            current_group = []
            recent_gaps = []  # 重置间隔记录

        current_group.append(segments[i])

    if current_group:
        result.append(current_group)

    return result