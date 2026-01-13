/**
 * 字幕数据处理模块
 * 与 Python 版本 (data.py) 保持完全一致的逻辑
 */

import type { SubtitleEntry } from '../types/index.js';

/**
 * 字幕数据容器类
 * 提供字幕类型检测和转换功能
 */
export class SubtitleData {
  private segments: SubtitleEntry[];

  constructor(segments: SubtitleEntry[]) {
    // 过滤空字幕并按时间排序
    this.segments = segments
      .filter(seg => seg.text && seg.text.trim())
      .sort((a, b) => a.startTime - b.startTime);
  }

  /**
   * 获取所有字幕片段
   */
  getSegments(): SubtitleEntry[] {
    return this.segments;
  }

  /**
   * 获取字幕数量
   */
  length(): number {
    return this.segments.length;
  }

  /**
   * 判断是否是单词级时间戳
   * 规则：
   * 1. 对于英文，每个segment应该只包含一个单词
   * 2. 对于中文，每个segment应该只包含一个汉字
   * 3. 允许20%的误差率
   *
   * 参考 Python 版本: data.py:56-76
   */
  isWordTimestamp(): boolean {
    if (this.segments.length === 0) {
      return false;
    }

    let validSegments = 0;
    const totalSegments = this.segments.length;

    for (const seg of this.segments) {
      const text = seg.text.trim();

      // 检查是否只包含一个英文单词或一个汉字
      // 使用更严格的 ≤2 字符检测（参考 VideoCaptioner 标准）
      const isAsciiWord = text.split(/\s+/).length === 1 && /^[\x00-\x7F]+$/.test(text);
      const isShortSegment = text.length <= 2;

      if (isAsciiWord || isShortSegment) {
        validSegments++;
      }
    }

    return (validSegments / totalSegments) >= 0.8;
  }

  /**
   * 将片段级别字幕转换为单词级别字幕，并按音素精确分配时间戳
   *
   * 这个方法借鉴了 VideoCaptioner 项目的实现策略，通过以下步骤处理：
   * 1. 使用多语言正则表达式识别所有有效字符和单词
   * 2. 基于音素理论分配时间戳（每4个字符=1个音素）
   * 3. 支持拉丁语系、中日韩、阿拉伯文、俄文等多种语言
   *
   * 优势：
   * - 时间戳分配比简单比例分配更准确
   * - 支持多语言混合文本
   * - 转换后可复用现有的批量断句框架
   *
   * 参考 Python 版本: data.py:78-164
   *
   * @returns 包含分割后字词级别 segments 的新 SubtitleData 实例
   */
  splitToWordSegments(): SubtitleData {
    const CHARS_PER_PHONEME = 4;  // 每个音素包含的字符数（基于语音学理论）
    const newSegments: SubtitleEntry[] = [];

    for (const seg of this.segments) {
      const text = seg.text;
      const duration = seg.endTime - seg.startTime;

      // 多语言字符匹配模式（借鉴 VideoCaptioner 的全面支持）
      // 分为两类：连续提取的语言和单字提取的语言
      const pattern = new RegExp(
        // 以单词形式出现的语言(连续提取)
        "[a-zA-Z\\u00c0-\\u00ff\\u0100-\\u017f']+" +  // 拉丁字母及其变体(英语、德语、法语等)
        "|[\\u0400-\\u04ff]+" +  // 西里尔字母(俄语等)
        "|[\\u0370-\\u03ff]+" +  // 希腊语
        "|[\\u0600-\\u06ff]+" +  // 阿拉伯语
        "|[\\u0590-\\u05ff]+" +  // 希伯来语
        "|\\d+" +  // 数字
        // 以单字形式出现的语言(单字提取)
        "|[\\u4e00-\\u9fff]" +  // 中文
        "|[\\u3040-\\u309f]" +  // 日文平假名
        "|[\\u30a0-\\u30ff]" +  // 日文片假名
        "|[\\uac00-\\ud7af]" +  // 韩文
        "|[\\u0e00-\\u0e7f][\\u0e30-\\u0e3a\\u0e47-\\u0e4e]*" +  // 泰文基字符及其音标组合
        "|[\\u0900-\\u097f]" +  // 天城文(印地语等)
        "|[\\u0980-\\u09ff]" +  // 孟加拉语
        "|[\\u0e80-\\u0eff]" +  // 老挝文
        "|[\\u1000-\\u109f]",   // 缅甸文
        'g'
      );

      const matches = Array.from(text.matchAll(pattern));

      if (matches.length === 0) {
        // 如果没有匹配到有效字符，跳过此段
        continue;
      }

      // 基于音素理论计算时间分配
      const totalPhonemes = matches.reduce((sum, match) => {
        return sum + Math.ceil(match[0].length / CHARS_PER_PHONEME);
      }, 0);

      const timePerPhoneme = duration / Math.max(totalPhonemes, 1);  // 防止除零错误

      // 为每个识别出的词/字符创建独立的时间戳
      let currentTime = seg.startTime;
      const MIN_WORD_DURATION = 50; // 最小时长50毫秒

      for (const match of matches) {
        const word = match[0];

        // 计算当前词的音素数量
        const wordPhonemes = Math.ceil(word.length / CHARS_PER_PHONEME);
        // 使用浮点数计算，确保每个单词至少有最小时长
        const wordDuration = Math.max(timePerPhoneme * wordPhonemes, MIN_WORD_DURATION);

        // 创建新的字词级 segment，确保时间不超出原始范围
        const wordEndTime = Math.min(currentTime + wordDuration, seg.endTime);

        newSegments.push({
          index: newSegments.length + 1,
          startTime: currentTime,
          endTime: wordEndTime,
          text: word,
        });

        currentTime = wordEndTime;
      }
    }

    return new SubtitleData(newSegments);
  }

  /**
   * 转换为纯文本格式
   * - 正确处理标点符号（不在标点前加空格）
   * - 保持单词之间的空格
   *
   * 参考 Python 版本: data.py:166-189
   */
  toText(): string {
    const texts: string[] = [];

    for (const seg of this.segments) {
      const text = seg.text.trim();

      if (!text) continue;

      // 如果是标点符号，不需要前导空格
      if (texts.length > 0 && /^[^\w\s]/.test(text)) {
        // 直接附加到前一个文本，不加空格
        texts[texts.length - 1] += text;
      } else {
        texts.push(text);
      }
    }

    return texts.join(' ');
  }
}
