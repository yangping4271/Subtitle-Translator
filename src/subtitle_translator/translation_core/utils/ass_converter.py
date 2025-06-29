import os
import re
import codecs

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

    subLines = re.sub(r'\d(\d:\d{2}:\d{2}),(\d{2})\d', r'\\1.\\2', subLines)
    subLines = re.sub(r'\s+-->\s+', ',', subLines)
    # replace style
    subLines = re.sub(r'<([ubi])>', r"{\\g<1>1}", subLines)
    subLines = re.sub(r'</([ubi])>', r"{\\g<1>0}", subLines)
    subLines = re.sub(r'<font\s+color="?#(\w{2})(\w{2})(\w{2})"?>', r"{\\c&H\\3\\2\\1&}", subLines)
    subLines = re.sub(r'</font>', "", subLines)
    return subLines
