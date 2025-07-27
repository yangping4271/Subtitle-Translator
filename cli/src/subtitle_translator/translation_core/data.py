import re
import os
from typing import List, Dict
from pathlib import Path
import logging

# 配置日志
logger = logging.getLogger("subtitle_translator_cli")

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
        filtered_segments = [seg for seg in segments if seg.text and seg.text.strip()]
        filtered_segments.sort(key=lambda x: x.start_time)
        self.segments = filtered_segments

    def __iter__(self):
        return iter(self.segments)
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def has_data(self) -> bool:
        """检查是否有字幕数据"""
        return len(self.segments) > 0
    
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
            
        valid_segments = 0
        total_segments = len(self.segments)
        
        for seg in self.segments:
            text = seg.text.strip()
            # 检查是否只包含一个英文单词或一个汉字
            if (len(text.split()) == 1 and text.isascii()) or len(text.strip()) <= 4:
                valid_segments += 1
        return (valid_segments / total_segments) >= 0.8

    def to_txt(self) -> str:
        """
        转换为纯文本格式
        - 正确处理标点符号（不在标点前加空格）
        - 保持单词之间的空格
        """
        # 过滤掉音效标记，并获取所有文本
        texts = []
        for seg in self.segments:
            text = seg.text.strip()
            # 如果是标点符号，不需要前导空格
            if text and not text[0].isalnum() and texts:
                texts[-1] = texts[-1].rstrip()
            texts.append(text)
        
        # 使用空格连接所有文本
        return ' '.join(texts).strip()

    def to_srt(self, save_path=None) -> str:
        """转换为SRT字幕格式"""
        srt_lines = []
        for n, seg in enumerate(self.segments, 1):
            srt_lines.append(f"{n}\n{seg.to_srt_ts()}\n{seg.transcript}\n")

        srt_text = "\n".join(srt_lines)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(srt_text)
        return srt_text

    def to_json(self) -> dict:
        """转换为JSON格式"""
        result_json = {}
        for i, segment in enumerate(self.segments, 1):
            # 检查是否有换行符
            if "\n" in segment.text:
                original_subtitle, translated_subtitle = segment.text.split("\n", 1)
            else:
                original_subtitle, translated_subtitle = segment.text, ""

            result_json[str(i)] = {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "original_subtitle": original_subtitle,
                "translated_subtitle": translated_subtitle
            }
        return result_json

    def merge_segments(self, start_index: int, end_index: int, merged_text: str = None):
        """合并从 start_index 到 end_index 的段（包含）"""
        if start_index < 0 or end_index >= len(self.segments) or start_index > end_index:
            raise IndexError("无效的段索引。")
        merged_start_time = self.segments[start_index].start_time
        merged_end_time = self.segments[end_index].end_time
        if merged_text is None:
            merged_text = ''.join(seg.text for seg in self.segments[start_index:end_index+1])
        merged_seg = SubtitleSegment(merged_text, merged_start_time, merged_end_time)
        # 替换 segments[start_index:end_index+1] 为 merged_seg
        self.segments[start_index:end_index+1] = [merged_seg]

    def merge_with_next_segment(self, index: int) -> None:
        """合并指定索引的段与下一个段"""
        if index < 0 or index >= len(self.segments) - 1:
            raise IndexError("索引超出范围或没有下一个段可合并。")
        current_seg = self.segments[index]
        next_seg = self.segments[index + 1]
        merged_text = f"{current_seg.text} {next_seg.text}"
        merged_seg = SubtitleSegment(merged_text, current_seg.start_time, next_seg.end_time)
        self.segments[index] = merged_seg
        # 删除下一个段
        del self.segments[index + 1]

    def save_translations(self, base_path: Path, translate_result: List[Dict], 
                        english_suffix: str = ".en.srt", target_lang_suffix: str = ".target.srt") -> None:
        """
        保存翻译结果，包括优化后的英文字幕和翻译后的目标语言字幕
        
        Args:
            base_path: 基础文件路径
            translate_result: 翻译结果列表
            english_suffix: 英文字幕文件后缀
            target_lang_suffix: 目标语言字幕文件后缀（默认为中文，但可以是任何语言）
        """
        # 构建输出文件路径
        base_name = base_path.stem
        output_dir = base_path.parent
        english_path = output_dir / f"{base_name}{english_suffix}"
        target_lang_path = output_dir / f"{base_name}{target_lang_suffix}"

        logger.info("开始保存...")

        # 保存优化后的英文字幕
        optimized_subtitles = {item["id"]: item["optimized"] for item in translate_result}
        self.save_translation(str(english_path), optimized_subtitles, "优化")

        # 保存翻译后的目标语言字幕
        translated_subtitles = {
            item["id"]: item.get("revised_translation", item["translation"])
            for item in translate_result
        }
        self.save_translation(str(target_lang_path), translated_subtitles, "翻译")

        # 只在最后统一打印总体统计
        total = len(self.segments)
        valid = sum(1 for item in translate_result if item.get("optimized", "").strip())
        skipped = total - valid
        logger.info(f"总字幕数: {total}, 有效字幕数: {valid}, 跳过字幕数: {skipped}")
        logger.info("保存完成")

    def save_translation(self, output_path: str, subtitle_dict: Dict[int, str], operation: str = "处理") -> None:
        """
        保存翻译或优化后的字幕文件
        """
        # 创建输出目录（如果不存在）
        output_dir = Path(output_path).parent
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # 生成SRT格式的字幕内容
        srt_lines = []
        logger.debug(f"{operation}字幕段落数: {len(self.segments)}")
        # logger.debug(f"字幕字典内容: {subtitle_dict}")
        
        # 记录有效字幕数
        valid_subtitle_count = 0
        
        for i, segment in enumerate(self.segments, 1):
            if i not in subtitle_dict:
                logger.warning(f"字幕 {i} 不在字典中")
                continue
                
            # 获取字幕内容，确保是字符串类型
            subtitle_text = subtitle_dict[i]
            if subtitle_text is None:
                logger.warning(f"字幕 {i} 的内容为None，将被跳过")
                continue
                
            processed_text = subtitle_text.strip()
                
            # 如果字幕内容为空，跳过该字幕
            if not processed_text:
                logger.debug(f"字幕 {i} 的内容为空，将被跳过")
                continue
                
            # 有效字幕数加1
            valid_subtitle_count += 1
            
            srt_lines.extend([
                str(valid_subtitle_count),  # 使用新的编号
                segment.to_srt_ts(),
                processed_text,
                ""  # 空行分隔
            ])

        # 写入文件
        srt_content = "\n".join(srt_lines)
        logger.debug(f"{operation}字幕内容:\n{srt_content}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        # 检查文件是否成功保存
        if not Path(output_path).exists():
            raise Exception(f"字幕{operation}失败: 文件未能成功保存")
            
        logger.info(f"{operation}后的字幕已保存至: {output_path}")

    def save_translations_to_files(self, translate_result: List[Dict], 
                                english_output: str, target_lang_output: str) -> None:
        """
        保存翻译结果到指定的文件路径
        
        Args:
            translate_result: 翻译结果列表
            english_output: 英文字幕输出路径
            target_lang_output: 目标语言字幕输出路径（可以是中文、日文、韩文等任何语言）
        """
        logger.info("开始保存...")

        # 保存优化后的英文字幕
        optimized_subtitles = {item["id"]: item["optimized"] for item in translate_result}
        self.save_translation(english_output, optimized_subtitles, "优化")

        # 保存翻译后的目标语言字幕
        translated_subtitles = {
            item["id"]: item.get("revised_translation", item["translation"])
            for item in translate_result
        }
        self.save_translation(target_lang_output, translated_subtitles, "翻译")

        # 只在最后统一打印总体统计
        total = len(self.segments)
        valid = sum(1 for item in translate_result if item.get("optimized", "").strip())
        skipped = total - valid
        logger.info(f"总字幕数: {total}, 有效字幕数: {valid}, 跳过字幕数: {skipped}")
        logger.info("保存完成")

    def __str__(self):
        return self.to_txt()


def load_subtitle(file_path: str) -> 'SubtitleData':
    """
    从文件加载字幕数据
    
    Args:
        file_path: 字幕文件路径，支持.srt格式
        
    Returns:
        SubtitleData: 解析后的字幕数据实例
        
    Raises:
        ValueError: 不支持的文件格式或文件读取错误
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 检查文件格式
    if not file_path.suffix.lower() == '.srt':
        raise ValueError("仅支持srt格式字幕文件")
        
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        content = file_path.read_text(encoding='gbk')
        
    return _parse_srt(content)

def _parse_srt(srt_str: str) -> 'SubtitleData':
    """
    解析SRT格式的字符串

    Args:
        srt_str: 包含SRT格式字幕的字符串
    Returns:
        SubtitleData: 解析后的字幕数据实例
    """
    segments = []
    srt_time_pattern = re.compile(
        r'(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})\s-->\s(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})'
    )
    blocks = re.split(r'\n\s*\n', srt_str.strip())

    # 如果超过90%的块都超过4行，说明可能包含翻译文本
    blocks_lines_count = [len(block.splitlines()) for block in blocks]
    if all(count <= 4 for count in blocks_lines_count) and sum(count == 4 for count in blocks_lines_count) / len(blocks_lines_count) > 0.9:
        has_translated_subtitle = True
    else:
        has_translated_subtitle = False

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue

        match = srt_time_pattern.match(lines[1])
        if not match:
            continue

        time_parts = list(map(int, match.groups()))
        start_time = sum([
            time_parts[0] * 3600000,
            time_parts[1] * 60000,
            time_parts[2] * 1000,
            time_parts[3]
        ])
        end_time = sum([
            time_parts[4] * 3600000,
            time_parts[5] * 60000,
            time_parts[6] * 1000,
            time_parts[7]
        ])

        if has_translated_subtitle:
            text = '\n'.join(lines[2:]).strip()
        else:
            text = ' '.join(lines[2:])

        segments.append(SubtitleSegment(text, start_time, end_time))

    return SubtitleData(segments)

def save_split_results(text: str, split_results: List[str], output_path: str) -> None:
    """
    保存原文本和断句结果到文件。
    
    Args:
        text: 原始文本
        split_results: 断句结果列表
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("原始文本:\n")
            f.write(text + "\n\n")
            f.write("断句结果:\n")
            for i, segment in enumerate(split_results):
                f.write(f"{segment}")
                if i < len(split_results) - 1:  # 确保不是最后一个分段
                    f.write("<br>")
        # 显示保存成功信息
        if os.path.exists(output_path):
            logger.info(f"断句结果已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存断句结果失败: {str(e)}")
        raise 