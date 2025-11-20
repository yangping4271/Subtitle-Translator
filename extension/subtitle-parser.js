/**
 * YouTube SubtitlePlus - 字幕解析器
 * ============================================
 * 提供统一的字幕文件解析功能
 * 支持格式：SRT、VTT、ASS
 *
 * 所有方法都是静态方法，可直接调用：
 * SubtitleParser.parseSRT(content)
 * SubtitleParser.parseVTT(content)
 * SubtitleParser.parseASS(content)
 */

class SubtitleParser {
  /**
   * 解析 SRT 格式字幕
   * @param {string} content - SRT 文件内容
   * @returns {Array<{startTime: number, endTime: number, text: string}>} 字幕数组
   */
  static parseSRT(content) {
    const subtitles = [];
    const blocks = content.trim().split(/\n\s*\n/);

    blocks.forEach(block => {
      const lines = block.trim().split('\n');
      if (lines.length >= 3) {
        const timeMatch = lines[1].match(/(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})/);
        if (timeMatch) {
          const startTime = this.parseTime(timeMatch[1], timeMatch[2], timeMatch[3], timeMatch[4]);
          const endTime = this.parseTime(timeMatch[5], timeMatch[6], timeMatch[7], timeMatch[8]);
          const text = lines.slice(2).join('\n').replace(/<[^>]*>/g, '');

          subtitles.push({ startTime, endTime, text });
        }
      }
    });

    return subtitles;
  }

  /**
   * 解析 VTT 格式字幕
   * @param {string} content - VTT 文件内容
   * @returns {Array<{startTime: number, endTime: number, text: string}>} 字幕数组
   */
  static parseVTT(content) {
    const subtitles = [];
    const lines = content.split('\n');
    let currentSubtitle = null;

    lines.forEach(line => {
      line = line.trim();

      if (line === 'WEBVTT' || line === '') return;

      const timeMatch = line.match(/(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})/);
      if (timeMatch) {
        if (currentSubtitle) {
          subtitles.push(currentSubtitle);
        }

        currentSubtitle = {
          startTime: this.parseTime(timeMatch[1], timeMatch[2], timeMatch[3], timeMatch[4]),
          endTime: this.parseTime(timeMatch[5], timeMatch[6], timeMatch[7], timeMatch[8]),
          text: ''
        };
      } else if (currentSubtitle && line) {
        currentSubtitle.text += (currentSubtitle.text ? '\n' : '') + line.replace(/<[^>]*>/g, '');
      }
    });

    if (currentSubtitle) {
      subtitles.push(currentSubtitle);
    }

    return subtitles;
  }

  /**
   * 解析 ASS 格式字幕（支持双语）
   * @param {string} content - ASS 文件内容
   * @returns {{english: Array, chinese: Array}} 包含英文和中文字幕的对象
   */
  static parseASS(content) {
    const result = { english: [], chinese: [] };
    const lines = content.split('\n');

    let inEventsSection = false;

    lines.forEach(line => {
      line = line.trim();

      // 检测Events部分开始
      if (line === '[Events]') {
        inEventsSection = true;
        return;
      }

      // 检测到新的段落，停止解析Events
      if (line.startsWith('[') && line !== '[Events]') {
        inEventsSection = false;
        return;
      }

      // 解析Dialogue行
      if (inEventsSection && line.startsWith('Dialogue:')) {
        const parts = line.split(',');
        if (parts.length >= 10) {
          const style = parts[3]; // Style name
          const startTime = this.parseASSTime(parts[1]); // Start time
          const endTime = this.parseASSTime(parts[2]); // End time

          // 提取文本内容，从第10个逗号后开始
          const textParts = parts.slice(9);
          let text = textParts.join(',').trim();

          // 清理ASS格式标签
          text = this.cleanASSText(text);

          if (text && startTime !== null && endTime !== null) {
            const subtitle = { startTime, endTime, text };

            // 根据Style分配到不同语言
            if (style === 'Default') {
              result.english.push(subtitle);
            } else if (style === 'Secondary') {
              result.chinese.push(subtitle);
            }
          }
        }
      }
    });

    return result;
  }

  /**
   * 解析 ASS 时间格式
   * @param {string} timeStr - ASS 时间字符串 (H:MM:SS.CC)
   * @returns {number|null} 时间（秒）或 null
   */
  static parseASSTime(timeStr) {
    const match = timeStr.match(/(\d+):(\d{2}):(\d{2})\.(\d{2})/);
    if (match) {
      const hours = parseInt(match[1]);
      const minutes = parseInt(match[2]);
      const seconds = parseInt(match[3]);
      const centiseconds = parseInt(match[4]);

      return hours * 3600 + minutes * 60 + seconds + centiseconds / 100;
    }
    return null;
  }

  /**
   * 清理 ASS 文本中的样式标签
   * @param {string} text - 原始文本
   * @returns {string} 清理后的文本
   */
  static cleanASSText(text) {
    return text
      .replace(/\{[^}]*\}/g, '') // 移除所有 {} 包围的标签
      .replace(/\\N/g, '\n') // 转换换行符
      .replace(/\\h/g, ' ') // 转换硬空格
      .trim();
  }

  /**
   * 解析通用时间格式
   * @param {string} hours - 小时
   * @param {string} minutes - 分钟
   * @param {string} seconds - 秒
   * @param {string} milliseconds - 毫秒
   * @returns {number} 时间（秒）
   */
  static parseTime(hours, minutes, seconds, milliseconds) {
    return parseInt(hours) * 3600 +
           parseInt(minutes) * 60 +
           parseInt(seconds) +
           parseInt(milliseconds) / 1000;
  }
}

// 导出到全局作用域（用于 content script 和 popup）
if (typeof window !== 'undefined') {
  window.SubtitleParser = SubtitleParser;
}
