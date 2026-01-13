/**
 * YouTube SubtitlePlus - 字幕解析器
 * ============================================
 * 提供统一的字幕文件解析功能
 * 支持格式：SRT、VTT、ASS
 */

import type { SimpleSubtitleEntry, ASSParseResult } from '../types';

export class SubtitleParser {
  /**
   * 解析 SRT 格式字幕
   */
  static parseSRT(content: string): SimpleSubtitleEntry[] {
    const subtitles: SimpleSubtitleEntry[] = [];
    const blocks = content.trim().split(/\n\s*\n/);

    blocks.forEach((block) => {
      const lines = block.trim().split('\n');
      if (lines.length >= 3) {
        const timeMatch = lines[1].match(
          /(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})/
        );
        if (timeMatch) {
          const startTime = this.parseTime(
            timeMatch[1],
            timeMatch[2],
            timeMatch[3],
            timeMatch[4]
          );
          const endTime = this.parseTime(
            timeMatch[5],
            timeMatch[6],
            timeMatch[7],
            timeMatch[8]
          );
          const text = lines
            .slice(2)
            .join('\n')
            .replace(/<[^>]*>/g, '');

          subtitles.push({ startTime, endTime, text });
        }
      }
    });

    return subtitles;
  }

  /**
   * 解析 VTT 格式字幕
   */
  static parseVTT(content: string): SimpleSubtitleEntry[] {
    const subtitles: SimpleSubtitleEntry[] = [];
    const lines = content.split('\n');
    let currentSubtitle: SimpleSubtitleEntry | null = null;

    lines.forEach((line) => {
      line = line.trim();

      if (line === 'WEBVTT' || line === '') return;

      const timeMatch = line.match(
        /(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})/
      );
      if (timeMatch) {
        if (currentSubtitle) {
          subtitles.push(currentSubtitle);
        }

        currentSubtitle = {
          startTime: this.parseTime(timeMatch[1], timeMatch[2], timeMatch[3], timeMatch[4]),
          endTime: this.parseTime(timeMatch[5], timeMatch[6], timeMatch[7], timeMatch[8]),
          text: '',
        };
      } else if (currentSubtitle && line) {
        currentSubtitle.text +=
          (currentSubtitle.text ? '\n' : '') + line.replace(/<[^>]*>/g, '');
      }
    });

    if (currentSubtitle) {
      subtitles.push(currentSubtitle);
    }

    return subtitles;
  }

  /**
   * 解析 ASS 格式字幕（支持双语）
   */
  static parseASS(content: string): ASSParseResult {
    const result: ASSParseResult = { english: [], chinese: [] };
    const lines = content.split('\n');

    let inEventsSection = false;

    lines.forEach((line) => {
      line = line.trim();

      if (line === '[Events]') {
        inEventsSection = true;
        return;
      }

      if (line.startsWith('[') && line !== '[Events]') {
        inEventsSection = false;
        return;
      }

      if (inEventsSection && line.startsWith('Dialogue:')) {
        const parts = line.split(',');
        if (parts.length >= 10) {
          const style = parts[3];
          const startTime = this.parseASSTime(parts[1]);
          const endTime = this.parseASSTime(parts[2]);

          const textParts = parts.slice(9);
          let text = textParts.join(',').trim();
          text = this.cleanASSText(text);

          if (text && startTime !== null && endTime !== null) {
            const subtitle: SimpleSubtitleEntry = { startTime, endTime, text };

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
   * 解析 ASS 时间格式（ASS 标准）
   * @param timeStr ASS 时间字符串 (H:MM:SS.CC)
   * @returns 时间（秒，浮点数）
   */
  static parseASSTime(timeStr: string): number | null {
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
   */
  static cleanASSText(text: string): string {
    return text
      .replace(/\{[^}]*\}/g, '')
      .replace(/\\N/g, '\n')
      .replace(/\\h/g, ' ')
      .trim();
  }

  /**
   * 解析通用时间格式（SRT 标准）
   * @returns 时间（秒，浮点数）
   */
  static parseTime(
    hours: string,
    minutes: string,
    seconds: string,
    milliseconds: string
  ): number {
    return (
      parseInt(hours) * 3600 +
      parseInt(minutes) * 60 +
      parseInt(seconds) +
      parseInt(milliseconds) / 1000
    );
  }
}

// 导出到全局作用域
declare global {
  interface Window {
    SubtitleParser: typeof SubtitleParser;
  }
}

if (typeof window !== 'undefined') {
  window.SubtitleParser = SubtitleParser;
}
