import os
import re
import codecs
from pathlib import Path
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def parse_srt_content(content: str) -> List[Dict]:
    """
    解析 SRT 内容为字幕条目列表

    使用空行分块策略，避免纯数字内容被误判为序号
    """
    # 标准化换行符
    content = content.replace('\r\n', '\n').replace('\r', '\n')

    # 按空行分块
    blocks = content.split('\n\n')
    subtitles = []

    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue

        # 第一行应该是序号
        if not lines[0].isdigit():
            continue

        # 第二行应该是时间戳
        timestamp_match = re.match(
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
            lines[1]
        )
        if not timestamp_match:
            continue

        # 剩余行是文本内容
        text = '\n'.join(lines[2:])

        subtitles.append({
            'id': lines[0],
            'start': timestamp_match.group(1),
            'end': timestamp_match.group(2),
            'text': text
        })

    return subtitles

def fix_timestamp_overlaps(subtitles: List[Dict]) -> tuple[List[Dict], int]:
    """
    修复字幕时间戳重叠，返回修复后的字幕和修复数量

    策略：
    - 如果 current_end > next_start，将 current_end 调整为 next_start
    - 如果调整后 current_end <= current_start，保留 1ms 间隔并记录警告
    """
    def time_to_ms(time_str: str) -> int:
        h, m, s_ms = time_str.split(':')
        s, ms = s_ms.split(',')
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

    def ms_to_time(ms: int) -> str:
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        ms %= 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    fixed_count = 0
    result = []

    for i, sub in enumerate(subtitles):
        current_start_ms = time_to_ms(sub['start'])
        current_end_ms = time_to_ms(sub['end'])

        # 检查是否需要修复
        if i < len(subtitles) - 1:
            next_start_ms = time_to_ms(subtitles[i + 1]['start'])

            if current_end_ms > next_start_ms:
                # 需要修复重叠
                new_end_ms = next_start_ms

                # 如果下一条开始时间 <= 当前开始时间，设为零时长以彻底消除重叠
                if new_end_ms <= current_start_ms:
                    new_end_ms = current_start_ms
                    logger.debug(
                        f"字幕 {sub['id']} 和 {subtitles[i+1]['id']} 开始时间相同或倒序，"
                        f"将字幕 {sub['id']} 设为零时长以消除重叠"
                    )

                sub = sub.copy()
                sub['end'] = ms_to_time(new_end_ms)
                fixed_count += 1

        result.append(sub)

    return result, fixed_count

def fileopen(input_file):
    encodings = ["utf-32", "utf-16", "utf-8", "cp1252", "gb2312", "gbk", "big5"]
    tmp = ''
    enc = 'utf-8'  # 默认编码
    for enc in encodings:
        try:
            with codecs.open(input_file, mode="r", encoding=enc) as fd:
                tmp = fd.read()
                break
        except Exception:
            continue
    return (tmp, enc)


def srt2ass_converter_func(input_file, pos):
    """
    将 SRT 文件转换为 ASS 格式的字幕行

    在内存中修复时间戳重叠，不修改原文件
    """
    if '.ass' in input_file:
        return input_file

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"{input_file} 不存在")

    src = fileopen(input_file)
    content = src[0]

    if '\ufeff' in content:
        content = content.replace('\ufeff', '')

    # 解析 SRT 内容
    subtitles = parse_srt_content(content)

    # 修复时间戳重叠
    subtitles, fixed_count = fix_timestamp_overlaps(subtitles)
    if fixed_count > 0:
        logger.info(f"{input_file}: 修复了 {fixed_count} 处时间戳重叠")

    # 转换为 ASS 格式
    subLines = ''
    for sub in subtitles:
        # 转换时间格式：00:00:10,240 -> 00:00:10.240
        start = sub['start'].replace(',', '.')
        end = sub['end'].replace(',', '.')
        text = sub['text']

        # 处理样式标签
        text = re.sub(r'<([ubi])>', r"{\\g<1>1}", text)
        text = re.sub(r'</([ubi])>', r"{\\g<1>0}", text)
        text = re.sub(r'<font\s+color="?#(\w{2})(\w{2})(\w{2})"?>', r"{\\c&H\3\2\1&}", text)
        text = re.sub(r'</font>', "", text)

        subLines += f'Dialogue: 0,{start},{end},{pos},,0,0,0,,{text}\n'

    return subLines


def convert_srt_to_ass(target_lang_srt_path: Path, english_srt_path: Path, output_dir: Path):
    """
    将目标语言字幕文件和英文字幕文件合并为双语ASS文件

    Args:
        target_lang_srt_path: 目标语言字幕文件路径（如日文、韩文、中文等）
        english_srt_path: 英文字幕文件路径
        output_dir: 输出目录

    Returns:
        Path: 生成的ASS文件路径
    """
    # 语言到字体的映射
    LANGUAGE_FONTS = {
        'zh': '宋体-简 黑体,11',            # 中文简体
        'zh-cn': '宋体-简 黑体,11',         # 中文简体
        'zh-tw': 'Noto Sans CJK TC,12',    # 中文繁体
        'ja': 'Noto Sans CJK JP,13',       # 日文  
        'ko': 'Noto Sans CJK KR,12',       # 韩文
        'fr': 'Noto Sans,14',              # 法文
        'de': 'Noto Sans,14',              # 德文
        'es': 'Noto Sans,14',              # 西班牙文
        'pt': 'Noto Sans,14',              # 葡萄牙文
        'ru': 'Noto Sans,13',              # 俄文
        'it': 'Noto Sans,14',              # 意大利文
        'ar': 'Noto Sans Arabic,13',       # 阿拉伯文
        'th': 'Noto Sans Thai,13',         # 泰文
        'vi': 'Noto Sans,13',              # 越南文
        'default': 'Noto Sans,13'          # 默认
    }
    
    # 从文件名检测语言
    def detect_language_from_filename(filepath):
        filename = Path(filepath).stem.lower()
        language_suffixes = ['zh-cn', 'zh-tw', 'zh', 'ja', 'ko', 'fr', 'de', 'es', 'pt', 'ru', 'it', 'ar', 'th', 'vi', 'en']
        for suffix in language_suffixes:
            if filename.endswith('.' + suffix):
                return suffix
        return 'default'
    
    # 检测目标语言并获取对应字体
    target_lang = detect_language_from_filename(target_lang_srt_path)
    target_font = LANGUAGE_FONTS.get(target_lang, LANGUAGE_FONTS['default'])
    
    head_str = f'''[Script Info]
; This is an Advanced Sub Station Alpha v4+ script.
Title:
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Serif,18,&H0000FFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,1,1,7,1
Style: Secondary,{target_font},&H0000FF00,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,1,1,7,1

[Events]
Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text'''

    # 目标语言字幕使用 Secondary 样式（显示在下方，绿色）
    target_lang_lines = srt2ass_converter_func(str(target_lang_srt_path), 'Secondary')

    # 英文字幕使用 Default 样式（显示在上方，青色）
    english_lines = srt2ass_converter_func(str(english_srt_path), 'Default')
    
    # 使用目标语言文件来获取编码信息和生成输出文件名
    src = fileopen(str(target_lang_srt_path))
    tmp = src[0]
    encoding = src[1]

    if u'\ufeff' in tmp:
        tmp = tmp.replace(u'\ufeff', '')

    # 移除语言后缀，支持多种语言代码格式
    base_name = target_lang_srt_path.stem
    # 移除常见的语言后缀模式，如 .zh, .ja, .en, .ko, .fr 等
    language_suffixes = ['.zh', '.zh-cn', '.zh-tw', '.ja', '.en', '.ko', '.fr', '.de', '.es', '.pt', '.ru', '.it', '.ar', '.th', '.vi']
    for suffix in language_suffixes:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    output_file = output_dir / f"{base_name}.ass"
    
    # 合并目标语言字幕和英文字幕
    output_str = head_str + '\n' + target_lang_lines + english_lines
    output_str = output_str.encode(encoding)

    with open(output_file, 'wb') as output:
        output.write(output_str)
    
    return output_file