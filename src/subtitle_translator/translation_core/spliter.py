import difflib
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List
from .split_by_llm import split_by_llm
from .data import SubtitleData, SubtitleSegment, save_split_results
from .config import get_default_config
from .utils.logger import setup_logger

logger = setup_logger("subtitle_spliter")

FIXED_NUM_THREADS = 1  # 固定的线程数量
SPLIT_RANGE = 30  # 在分割点前后寻找最大时间间隔的范围
MAX_GAP = 1500  # 允许每个词语之间的最大时间间隔 ms

class SubtitleProcessError(Exception):
    """字幕处理相关的异常"""
    pass

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


def merge_segments_based_on_sentences(segments: List[SubtitleSegment], sentences: List[str], max_unmatched: int = 5) -> List[SubtitleSegment]:
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
    asr_index = 0  # 当前分段索引位置
    threshold = 0.5  # 相似度阈值
    max_shift = 30  # 滑动窗口的最大偏移量
    unmatched_count = 0  # 未匹配句子计数

    new_segments = []

    for sentence in sentences:
        sentence_proc = preprocess_text(sentence)
        word_count = count_words(sentence_proc)
        best_ratio = 0.0
        best_pos = None
        best_window_size = 0

        # 滑动窗口大小，优先考虑接近句子词数的窗口
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
                    break  # 完全匹配
            if best_ratio == 1.0:
                break  # 完全匹配

        if best_ratio >= threshold and best_pos is not None:
            start_seg_index = best_pos
            end_seg_index = best_pos + best_window_size - 1
            
            segs_to_merge = segments[start_seg_index:end_seg_index + 1]

            # 按照时间切分避免合并跨度大的
            seg_groups = merge_by_time_gaps(segs_to_merge, max_gap=MAX_GAP)

            for group in seg_groups:
                # 直接使用LLM返回的原始句子，完全保留格式和标点
                merged_text = sentence_proc
                
                merged_start_time = group[0].start_time
                merged_end_time = group[-1].end_time
                merged_seg = SubtitleSegment(merged_text, merged_start_time, merged_end_time)
                
                new_segments.append(merged_seg)
            
            max_shift = 30
            asr_index = end_seg_index + 1  # 移动到下一个未处理的分段
        else:
            logger.warning(f"无法匹配句子: {sentence}")
            unmatched_count += 1
            if unmatched_count > max_unmatched:
                logger.error(f"未匹配句子数量超过阈值 ({max_unmatched})，返回原始分段")
                return segments
            max_shift = 100
            asr_index = min(asr_index + 1, asr_len - 1)  # 确保不会超出范围
    
    # 如果没有成功匹配任何句子，返回原始分段
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


def process_by_llm(segments: List[SubtitleSegment], 
                   model: str = None,
                   max_word_count_english: int = None,
                   batch_index: int = None) -> List[SubtitleSegment]:
    """
    使用LLM处理分段
    
    Args:
        segments: 字幕分段列表
        model: 使用的语言模型，如果为None则使用配置中的断句模型
        max_word_count_english: 英文最大单词数
        batch_index: 批次编号
        
    Returns:
        List[SubtitleSegment]: 处理后的字幕分段列表
    """
    config = get_default_config()
    max_word_count_english = max_word_count_english or config.max_word_count_english
    
    # 如果没有指定模型，使用配置中的断句模型
    if model is None:
        model = config.split_model
        
    # 修改合并文本的方式，添加空格
    txt = " ".join([seg.text.strip() for seg in segments])
    # 记录当前批次的单词数
    current_words = count_words(txt)
    logger.info(f"批次 {batch_index}: 处理文本单词数: {current_words}")
    
    # 使用LLM拆分句子
    sentences = split_by_llm(txt, 
                           model=model, 
                           max_word_count_english=max_word_count_english)
    logger.info(f"批次 {batch_index}: 句子提取完成，共 {len(sentences)} 句")
    # 对当前分段进行合并处理
    merged_segments = merge_segments_based_on_sentences(segments, sentences)
    return merged_segments


def split_by_sentences(asr_data: SubtitleData, word_threshold: int = 500) -> List[SubtitleData]:
    """
    根据句号等标点符号切分句子，并按指定单词数阈值分组
    
    Args:
        asr_data: 字幕数据
        word_threshold: 每组最大单词数，默认500
        
    Returns:
        List[SubtitleData]: 按单词数阈值分组后的字幕数据列表
    """
    # 定义句子结束标志
    sentence_end_markers = ['.', '!', '?', '。', '！', '？', '…']
    # 定义分句标点
    split_markers = [',', '，', ';', '；', '、']
    
    # 预处理字幕数据
    segments = preprocess_segments(asr_data.segments)
    
    # 按句子切分
    sentence_segments = []
    current_sentence_segments = []
    
    for seg in segments:
        current_sentence_segments.append(seg)
        text = seg.text.strip()
        
        # 检查是否是句子结尾
        if any(text.endswith(marker) for marker in sentence_end_markers):
            if current_sentence_segments:
                sentence_segments.append(current_sentence_segments)
                current_sentence_segments = []
    
    # 处理最后一组未完成的句子
    if current_sentence_segments:
        sentence_segments.append(current_sentence_segments)
    
    # 按单词数阈值分组
    batched_data = []
    current_batch = []
    current_segments = []
    current_word_count = 0
    
    def split_long_sentence(sentence_segs: List[SubtitleSegment]) -> List[List[SubtitleSegment]]:
        """拆分过长的句子"""
        result = []
        temp_segs = []
        temp_word_count = 0
        
        for seg in sentence_segs:
            seg_text = seg.text.strip()
            seg_word_count = count_words(seg_text)
            
            # 如果当前段落加上之前的已经超过阈值，并且当前段落以分句标点结尾
            if (temp_word_count + seg_word_count > word_threshold and 
                any(seg_text.endswith(marker) for marker in split_markers)):
                if temp_segs:
                    result.append(temp_segs)
                    temp_segs = []
                    temp_word_count = 0
            
            temp_segs.append(seg)
            temp_word_count += seg_word_count
            
            # 如果累积的单词数已经接近阈值，强制分段
            if temp_word_count >= word_threshold * 1.2:
                if temp_segs:
                    result.append(temp_segs)
                    temp_segs = []
                    temp_word_count = 0
        
        # 处理剩余的段落
        if temp_segs:
            result.append(temp_segs)
        
        return result
    
    for sentence in sentence_segments:
        # 计算当前句子的单词数
        sentence_text = " ".join([seg.text for seg in sentence])
        sentence_word_count = count_words(sentence_text)
        
        # 如果当前句子超过阈值，尝试拆分
        if sentence_word_count >= word_threshold:
            # 先保存当前批次
            if current_segments:
                batched_data.append(SubtitleData(current_segments))
                current_batch = []
                current_segments = []
                current_word_count = 0
            
            # 拆分长句子
            split_parts = split_long_sentence(sentence)
            for part in split_parts:
                batched_data.append(SubtitleData(part))
            continue
            
        # 如果添加当前句子后超过阈值，先保存当前批次，然后开始新批次
        if current_word_count + sentence_word_count > word_threshold and current_segments:
            batched_data.append(SubtitleData(current_segments))
            current_batch = []
            current_segments = []
            current_word_count = 0
        
        current_batch.append(sentence)
        current_segments.extend(sentence)
        current_word_count += sentence_word_count
    
    # 处理最后一批未满的数据
    if current_segments:
        batched_data.append(SubtitleData(current_segments))
    
    return batched_data


def merge_segments(asr_data: SubtitleData, 
                   model: str = None, 
                   num_threads: int = FIXED_NUM_THREADS, 
                   save_split: str = None) -> SubtitleData:
    """
    合并字幕分段
    
    Args:
        asr_data: 字幕数据
        model: 使用的语言模型，如果为None则使用配置中的断句模型
        num_threads: 线程数量
        save_split: 保存断句结果的文件路径
    """
    
    # 如果没有指定模型，使用配置中的断句模型
    if model is None:
        config = get_default_config()
        model = config.split_model
    
    # 预处理字幕数据，移除纯标点符号的分段，并处理仅包含字母和撇号的文本
    asr_data.segments = preprocess_segments(asr_data.segments)
    
    # 使用新的按单词数分组方法
    word_threshold = 500
    asr_data_segments = split_by_sentences(asr_data, word_threshold=word_threshold)
    total_segments = len(asr_data_segments)
    logger.info(f"按每组{word_threshold}个单词分组，共 {total_segments} 批次")

    # 检查每个批次的单词数
    for i, segment in enumerate(asr_data_segments):
        text = " ".join([seg.text.strip() for seg in segment.segments])
        word_count = count_words(text)
        logger.info(f"批次 {i+1}/{total_segments}: 单词数 {word_count}")

    # 多线程处理每个分段
    logger.info("开始并行处理每个分段...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        def process_segment(args):
            index, asr_data_part = args
            try:
                return process_by_llm(asr_data_part.segments, model=model, batch_index=index+1)
            except Exception as e:
                raise Exception(f"批次 {index+1} LLM处理失败: {str(e)}")

        # 并行处理所有分段，添加批次编号
        processed_segments = list(executor.map(process_segment, enumerate(asr_data_segments)))

    # 合并所有处理后的分段
    final_segments = []
    for segment in processed_segments:
        final_segments.extend(segment)

    final_segments.sort(key=lambda seg: seg.start_time)

    # 如果需要保存断句结果
    if save_split:
        try:
            # 获取所有文本
            all_text = asr_data.to_txt()
            # 获取所有处理后的分段文本
            all_segments = [seg.text for seg in final_segments]
            
            # 显示断句结果
            logger.info(f"所有分段断句完成，共 {len(all_segments)} 句")
            for i, segment in enumerate(all_segments, 1):
                logger.debug(f"第 {i} 句: {segment}")
            
            # 保存结果
            # save_split_results(all_text, all_segments, save_split)

        except Exception as e:
            logger.error(f"保存断句结果失败: {str(e)}")

    merge_short_segment(final_segments)

    # 创建最终的字幕数据对象
    final_asr_data = SubtitleData(final_segments)
    return final_asr_data