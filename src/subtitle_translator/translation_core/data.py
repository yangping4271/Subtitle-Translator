import re
import math
from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass
import logging

# 配置日志
logger = logging.getLogger("subtitle_translator_cli")

# 常量定义
CHARS_PER_PHONEME = 4  # 每个音素包含的字符数（基于语音学理论）
ELLIPSIS_PLACEHOLDER = "<<<ELLIPSIS>>>"
WORD_TIMESTAMP_THRESHOLD = 0.8  # 单词级时间戳判定阈值
MAX_CHAR_LENGTH_FOR_WORD = 2  # 单词级时间戳的最大字符长度


@dataclass
class PreSplitSentence:
    """预分句数据结构（移植自 youtube-subtitle）"""

    text: str
    word_start_index: int
    word_end_index: int
    start_time: int
    end_time: int


def normalize_chinese_punctuation(text: str) -> str:
    """保留中文句内标点，删除行尾弱标点，并补齐中英文/数字间空格。"""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fff])", r"\1 \2", text)
    text = re.sub(r"[，,、。．.；;：:]+$", "", text)
    return text


def normalize_english_punctuation(text: str) -> str:
    """按 Netflix 规范处理英文标点：保留 ? ! ... ' "，删除 . , ; :"""
    # 先保护省略号
    text = text.replace("...", ELLIPSIS_PLACEHOLDER)

    # 删除 . , ; :
    text = re.sub(r"[.,;:]", "", text)

    # 恢复省略号
    text = text.replace(ELLIPSIS_PLACEHOLDER, "...")
    return text


class SubtitleSegment:
    """单个字幕段的数据结构"""

    def __init__(self, text: str, start_time: int, end_time: int):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time

    def to_srt_ts(self) -> str:
        """转换为SRT时间戳格式"""
        return f"{self._ms_to_srt_time(self.start_time)} --> {self._ms_to_srt_time(self.end_time)}"

    @staticmethod
    def _ms_to_srt_time(ms: int) -> str:
        """将毫秒转换为SRT时间格式 (HH:MM:SS,mmm)"""
        total_seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

    @property
    def transcript(self) -> str:
        """返回字幕文本"""
        return self.text

    def __str__(self) -> str:
        return f"SubtitleSegment({self.text}, {self.start_time}, {self.end_time})"


class SubtitleData:
    """字幕数据的主要容器类"""

    def __init__(self, segments: List[SubtitleSegment]):
        # 去除 segments.text 为空的
        self.segments = sorted(
            (seg for seg in segments if seg.text and seg.text.strip()),
            key=lambda seg: seg.start_time,
        )

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def is_word_timestamp(self) -> bool:
        """
        判断是否是字级时间戳
        规则：
        1. 对于英文，每个segment应该只包含一个单词
        2. 对于中文，每个segment应该只包含一个汉字
        3. 允许20%的误差率
        """
        if not self.segments:
            return False

        valid_segments = sum(
            1 for seg in self.segments if self._is_single_word_or_char(seg.text.strip())
        )
        return (valid_segments / len(self.segments)) >= WORD_TIMESTAMP_THRESHOLD

    def _is_single_word_or_char(self, text: str) -> bool:
        """检查是否只包含一个英文单词或一个汉字"""
        return (len(text.split()) == 1 and text.isascii()) or len(
            text.strip()
        ) <= MAX_CHAR_LENGTH_FOR_WORD

    def split_to_word_segments(self) -> "SubtitleData":
        """按语言拆分字词，以字符权重分配原片段的时间。"""
        new_segments = []
        for seg in self.segments:
            text = seg.text
            duration = seg.end_time - seg.start_time

            # 多语言字符匹配模式（借鉴VideoCaptioner的全面支持）
            # 分为两类：连续提取的语言和单字提取的语言
            pattern = (
                # 以单词形式出现的语言(连续提取)，包括附着的标点符号
                r"[a-zA-Z\u00c0-\u00ff\u0100-\u017f']+[.,!?;:]*"  # 拉丁字母及其变体(英语、德语、法语等)
                r"|[\u0400-\u04ff]+"  # 西里尔字母(俄语等)
                r"|[\u0370-\u03ff]+"  # 希腊语
                r"|[\u0600-\u06ff]+"  # 阿拉伯语
                r"|[\u0590-\u05ff]+"  # 希伯来语
                r"|\d+[.,]*"  # 数字（可能带小数点或逗号）
                # 以单字形式出现的语言(单字提取)
                r"|[\u4e00-\u9fff]"  # 中文
                r"|[\u3040-\u309f]"  # 日文平假名
                r"|[\u30a0-\u30ff]"  # 日文片假名
                r"|[\uac00-\ud7af]"  # 韩文
                r"|[\u0e00-\u0e7f][\u0e30-\u0e3a\u0e47-\u0e4e]*"  # 泰文基字符及其音标组合
                r"|[\u0900-\u097f]"  # 天城文(印地语等)
                r"|[\u0980-\u09ff]"  # 孟加拉语
                r"|[\u0e80-\u0eff]"  # 老挝文
                r"|[\u1000-\u109f]"  # 缅甸文
            )

            words = re.findall(pattern, text)
            if not words:
                continue
            phonemes = [math.ceil(len(word) / CHARS_PER_PHONEME) for word in words]
            time_per_phoneme = duration / sum(phonemes)
            current_time = seg.start_time
            for word, weight in zip(words, phonemes):
                word_end_time = min(
                    current_time + int(time_per_phoneme * weight), seg.end_time
                )
                new_segments.append(SubtitleSegment(word, current_time, word_end_time))
                current_time = word_end_time

        return SubtitleData(new_segments)

    def to_txt(self) -> str:
        """按时间顺序连接字幕文本。"""
        return " ".join(seg.text.strip() for seg in self.segments).strip()

    def to_json(self) -> dict:
        """转换为JSON格式"""
        result_json = {}
        for i, segment in enumerate(self.segments, 1):
            original_subtitle, _, translated_subtitle = segment.text.partition("\n")
            result_json[str(i)] = {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "original_subtitle": original_subtitle,
                "translated_subtitle": translated_subtitle,
            }
        return result_json

    def save_translation(
        self, output_path: str, subtitle_dict: Dict[int, str], operation: str = "处理"
    ) -> None:
        """
        保存翻译或优化后的字幕文件

        Args:
            output_path: 输出文件路径
            subtitle_dict: 字幕字典
            operation: 操作类型（"优化" 或 "翻译"）
        """
        # 创建输出目录（如果不存在）
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成SRT格式的字幕内容
        srt_lines = []
        logger.info(f"{operation}字幕段落数: {len(self.segments)}")

        # 记录写入字幕数
        saved_subtitle_count = 0
        empty_translation_count = 0

        for i, segment in enumerate(self.segments, 1):
            if i not in subtitle_dict:
                logger.warning(f"字幕 {i} 不在字典中")
                continue

            processed_text = (subtitle_dict[i] or "").strip()
            if operation == "翻译":
                processed_text = normalize_chinese_punctuation(processed_text)
            elif operation == "优化":
                processed_text = normalize_english_punctuation(processed_text)
                if not processed_text:
                    logger.warning(f"字幕 {i} 的优化内容为空，将回退为原文")
                    processed_text = normalize_english_punctuation(segment.text.strip())

            if operation == "翻译" and not processed_text:
                empty_translation_count += 1
                logger.info(f"字幕 {i} 的翻译为空，保留时间轴并写入空字幕")

            saved_subtitle_count += 1

            srt_lines.extend(
                [
                    str(saved_subtitle_count),  # 使用新的编号
                    segment.to_srt_ts(),
                    processed_text,
                    "",  # 空行分隔
                ]
            )

        # 写入文件
        Path(output_path).write_text("\n".join(srt_lines), encoding="utf-8")

        logger.info(f"{operation}后的字幕已保存至: {output_path}")
        if operation == "翻译" and empty_translation_count > 0:
            logger.info(f"空翻译字幕数: {empty_translation_count}")

    def save_translations_to_files(
        self, translate_result: List[Dict], english_output: str, target_lang_output: str
    ) -> None:
        """
        保存翻译结果到指定的文件路径

        Args:
            translate_result: 翻译结果列表
            english_output: 英文字幕输出路径
            target_lang_output: 目标语言字幕输出路径（可以是中文、日文、韩文等任何语言）
        """
        logger.info("开始保存...")

        # 保存优化后的英文字幕
        optimized_subtitles = {
            item["id"]: item["optimized"] for item in translate_result
        }
        self.save_translation(english_output, optimized_subtitles, "优化")

        # 保存翻译后的目标语言字幕
        translated_subtitles = {
            item["id"]: item["translation"] for item in translate_result
        }
        self.save_translation(target_lang_output, translated_subtitles, "翻译")

        # 只在最后统一打印总体统计
        total = len(self.segments)
        english_fallback = sum(
            1 for item in translate_result if not (item.get("optimized") or "").strip()
        )
        empty_translations = sum(
            1
            for item in translate_result
            if not (item.get("translation") or "").strip()
        )
        logger.info(
            f"总字幕数: {total}, 英文回退数: {english_fallback}, 空翻译数: {empty_translations}"
        )
        logger.info("保存完成")

    def __str__(self):
        return self.to_txt()


def load_subtitle(file_path: str) -> "SubtitleData":
    """
    从文件加载字幕数据

    Args:
        file_path: 字幕文件路径，支持.srt格式

    Returns:
        SubtitleData: 解析后的字幕数据实例

    Raises:
        ValueError: 不支持的文件格式或文件读取错误
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    # 检查文件格式
    if not path.suffix.lower() == ".srt":
        raise ValueError("仅支持srt格式字幕文件")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="gbk")

    return _parse_srt(content)


def _validate_timestamps(segments: List[SubtitleSegment]) -> None:
    """验证字幕时间戳合法性，发现问题时抛出异常"""
    from ..exceptions import SubtitleProcessError

    # 检查0：单条字幕 end_time < start_time（对所有数量都检查）
    for i, seg in enumerate(segments, 1):
        if seg.end_time < seg.start_time:
            raise SubtitleProcessError(
                f"SRT 时间戳不合法：第 {i} 条字幕结束时间 "
                f"({SubtitleSegment._ms_to_srt_time(seg.end_time)}) "
                f"早于开始时间 "
                f"({SubtitleSegment._ms_to_srt_time(seg.start_time)})",
                suggestion="请检查 SRT 文件时间戳是否正确。",
            )

    if len(segments) < 2:
        return

    # 检查1：时间戳倒退
    for i in range(1, len(segments)):
        if segments[i].start_time < segments[i - 1].start_time:
            raise SubtitleProcessError(
                f"SRT 时间戳不合法：第 {i + 1} 条字幕开始时间 "
                f"({SubtitleSegment._ms_to_srt_time(segments[i].start_time)}) "
                f"早于第 {i} 条 "
                f"({SubtitleSegment._ms_to_srt_time(segments[i - 1].start_time)})",
                suggestion="请检查 SRT 文件时间戳是否单调递增。",
            )

    # 检查2：过多重复开始时间（超过 20% 的相邻字幕共享同一开始时间）
    same_start_count = sum(
        1
        for i in range(1, len(segments))
        if segments[i].start_time == segments[i - 1].start_time
    )
    ratio = same_start_count / (len(segments) - 1)
    if ratio > 0.2:
        raise SubtitleProcessError(
            f"SRT 时间戳不合法：{same_start_count} 对相邻字幕共享相同开始时间"
            f"（占比 {ratio:.0%}），时间戳可能已损坏。",
            suggestion="请检查 SRT 文件的时间戳是否正确。",
        )


def _parse_srt(srt_str: str) -> "SubtitleData":
    """
    解析SRT格式的字符串

    Args:
        srt_str: 包含SRT格式字幕的字符串
    Returns:
        SubtitleData: 解析后的字幕数据实例
    """
    segments = []
    srt_time_pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})\s-->\s(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})"
    )
    blocks = re.split(r"\n\s*\n", srt_str.strip())

    # 绝大多数块恰有两行文本时，保留双语字幕的换行。
    line_counts = [len(block.splitlines()) for block in blocks]
    has_translated_subtitle = (
        all(count <= 4 for count in line_counts)
        and sum(count == 4 for count in line_counts) / len(line_counts) > 0.9
    )

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue

        # 灵活查找时间戳行，而不是固定在 lines[1]
        match = None
        time_line_index = -1
        for i, line in enumerate(lines):
            match = srt_time_pattern.match(line)
            if match:
                time_line_index = i
                break

        if not match or time_line_index == -1:
            continue

        time_parts = list(map(int, match.groups()))
        start_time, end_time = (
            sum(value * scale for value, scale in zip(parts, (3600000, 60000, 1000, 1)))
            for parts in (time_parts[:4], time_parts[4:])
        )
        separator = "\n" if has_translated_subtitle else " "
        text = separator.join(lines[time_line_index + 1 :])
        if has_translated_subtitle:
            text = text.strip()

        segments.append(SubtitleSegment(text, start_time, end_time))

    _validate_timestamps(segments)
    return SubtitleData(segments)
