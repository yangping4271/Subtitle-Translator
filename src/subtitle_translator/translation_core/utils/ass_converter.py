import os
import re
import codecs
from pathlib import Path

def fileopen(input_file):
    encodings = ["utf-32", "utf-16", "utf-8", "cp1252", "gb2312", "gbk", "big5"]
    tmp = ''
    for enc in encodings:
        try:
            with codecs.open(input_file, mode="r", encoding=enc) as fd:
                tmp = fd.read()
                break
        except:
            # print enc + ' failed'
            continue
    return [tmp, enc]


def srt2ass_converter_func(input_file, pos):
    if '.ass' in input_file:
        return input_file

    if not os.path.isfile(input_file):
        print(input_file + ' not exist')
        return
    
    src = fileopen(input_file)
    tmp = src[0]

    if u'\ufeff' in tmp:
        tmp = tmp.replace(u'\ufeff', '')
    
    tmp = tmp.replace("\r", "")
    lines = [x.strip() for x in tmp.split("\n") if x.strip()]
    subLines = ''
    tmpLines = ''
    lineCount = 0
    output_file = '.'.join(input_file.split('.')[:-1])
    output_file += '.ass'

    for ln in range(len(lines)):
        line = lines[ln]
        if line.isdigit() and re.match(r'-?\d\d:\d\d:\d\d', lines[(ln+1)]):
            if tmpLines:
                subLines += tmpLines + "\n"
            tmpLines = ''
            lineCount = 0
            continue
        else:
            if re.match(r'-?\d\d:\d\d:\d\d', line):
                line = line.replace('-0', '0')
                tmpLines += 'Dialogue: 0,' + line + ','+ pos + ',,0,0,0,,'
            else:
                if lineCount < 2:
                    tmpLines += line
                else:
                    tmpLines += "\n" + line
            lineCount += 1
        ln += 1


    subLines += tmpLines + "\n"

    subLines = re.sub(r'(\d{2}:\d{2}:\d{2}),(\d{3})', r'\1.\2', subLines)
    subLines = re.sub(r'\s+-->\s+', ',', subLines)
    # replace style
    subLines = re.sub(r'<([ubi])>', r"{\\g<1>1}", subLines)
    subLines = re.sub(r'</([ubi])>', r"{\\g<1>0}", subLines)
    subLines = re.sub(r'<font\s+color="?#(\w{2})(\w{2})(\w{2})"?>', r"{\\c&H\\3\\2\\1&}", subLines)
    subLines = re.sub(r'</font>', "", subLines)
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
    
    tmp = tmp.replace("", "")
    lines = [x.strip() for x in tmp.split("\n") if x.strip()]
    
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