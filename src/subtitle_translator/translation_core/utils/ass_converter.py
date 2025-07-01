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

def convert_srt_to_ass(zh_srt_path: Path, en_srt_path: Path, output_dir: Path):
    head_str = '''[Script Info]
; This is an Advanced Sub Station Alpha v4+ script.
Title:
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Serif,18,&H0000FFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,1,1,7,1
Style: Secondary,宋体-简 黑体,11,&H0000FF00,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,1,1,7,1

[Events]
Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text'''

    subLines1 = srt2ass_converter_func(str(zh_srt_path), 'Secondary')
    subLines2 = srt2ass_converter_func(str(en_srt_path), 'Default')
    
    src = fileopen(str(zh_srt_path))
    tmp = src[0]
    encoding = src[1]

    if u'\ufeff' in tmp:
        tmp = tmp.replace(u'\ufeff', '')
    
    tmp = tmp.replace("", "")
    lines = [x.strip() for x in tmp.split("\n") if x.strip()]
    
    # 移除语言后缀，支持多种语言代码格式
    base_name = zh_srt_path.stem
    # 移除常见的语言后缀模式，如 .zh, .ja, .en, .ko, .fr 等
    language_suffixes = ['.zh', '.zh-cn', '.zh-tw', '.ja', '.en', '.ko', '.fr', '.de', '.es', '.pt', '.ru', '.it', '.ar', '.th', '.vi']
    for suffix in language_suffixes:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    output_file = output_dir / f"{base_name}.ass"
    
    output_str = head_str + '\n' + subLines1 + subLines2
    output_str = output_str.encode(encoding)

    with open(output_file, 'wb') as output:
        output.write(output_str)
    
    return output_file