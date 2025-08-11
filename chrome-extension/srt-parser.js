// SRT字幕文件解析工具
class SRTParser {
  /**
   * 解析SRT格式的字幕文本
   * @param {string} srtContent - SRT文件内容
   * @returns {Array} 字幕段落数组
   */
  static parse(srtContent) {
    if (!srtContent || typeof srtContent !== 'string') {
      return [];
    }

    const subtitles = [];
    
    // 标准化换行符并分割段落
    const blocks = srtContent
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n')
      .split(/\n\s*\n/)
      .filter(block => block.trim());

    for (const block of blocks) {
      const lines = block.trim().split('\n');
      
      if (lines.length < 3) {
        continue; // 跳过不完整的段落
      }

      // 第一行: 序号
      const index = parseInt(lines[0].trim());
      if (isNaN(index)) {
        continue;
      }

      // 第二行: 时间轴
      const timeMatch = lines[1].match(/(\d{2}:\d{2}:\d{2}[,.]?\d{0,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]?\d{0,3})/);
      if (!timeMatch) {
        continue;
      }

      const startTime = this.parseTime(timeMatch[1]);
      const endTime = this.parseTime(timeMatch[2]);

      // 第三行及之后: 字幕文本
      const text = lines.slice(2)
        .join('\n')
        .replace(/<[^>]*>/g, '') // 移除HTML标签
        .trim();

      if (text) {
        subtitles.push({
          index,
          startTime,
          endTime,
          text,
          originalText: text // 保存原始文本
        });
      }
    }

    return subtitles.sort((a, b) => a.startTime - b.startTime);
  }

  /**
   * 解析时间戳为秒数
   * @param {string} timeStr - 时间字符串 (HH:MM:SS,mmm 或 HH:MM:SS.mmm)
   * @returns {number} 秒数
   */
  static parseTime(timeStr) {
    const match = timeStr.match(/(\d{2}):(\d{2}):(\d{2})[,.]?(\d{0,3})/);
    if (!match) {
      return 0;
    }

    const hours = parseInt(match[1]);
    const minutes = parseInt(match[2]);
    const seconds = parseInt(match[3]);
    const milliseconds = parseInt((match[4] || '0').padEnd(3, '0'));

    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000;
  }

  /**
   * 根据时间查找对应的字幕
   * @param {Array} subtitles - 字幕数组
   * @param {number} currentTime - 当前时间（秒）
   * @returns {Object|null} 匹配的字幕对象
   */
  static findSubtitleAtTime(subtitles, currentTime) {
    if (!Array.isArray(subtitles) || subtitles.length === 0) {
      return null;
    }

    return subtitles.find(subtitle => 
      currentTime >= subtitle.startTime && currentTime <= subtitle.endTime
    ) || null;
  }

  /**
   * 合并双语字幕数组
   * @param {Array} originalSubtitles - 原文字幕数组
   * @param {Array} translatedSubtitles - 翻译字幕数组
   * @returns {Array} 合并后的双语字幕数组
   */
  static mergeBilingualSubtitles(originalSubtitles, translatedSubtitles) {
    if (!Array.isArray(originalSubtitles) || !Array.isArray(translatedSubtitles)) {
      return originalSubtitles || translatedSubtitles || [];
    }

    const mergedSubtitles = [];
    
    // 创建多种匹配策略的映射
    const translationMaps = {
      exact: new Map(),    // 精确时间匹配
      index: new Map(),    // 索引匹配
      fuzzy: new Map()     // 模糊时间匹配
    };
    
    translatedSubtitles.forEach((subtitle, index) => {
      // 精确时间匹配
      const exactKey = `${subtitle.startTime.toFixed(3)}-${subtitle.endTime.toFixed(3)}`;
      translationMaps.exact.set(exactKey, subtitle.text);
      
      // 索引匹配
      translationMaps.index.set(index, subtitle.text);
      
      // 模糊时间匹配（允许100ms误差）
      const fuzzyKey = `${Math.round(subtitle.startTime * 10)}-${Math.round(subtitle.endTime * 10)}`;
      translationMaps.fuzzy.set(fuzzyKey, subtitle.text);
    });

    // 匹配原文和翻译
    originalSubtitles.forEach((original, index) => {
      let translation = null;
      
      // 1. 尝试精确时间匹配
      const exactKey = `${original.startTime.toFixed(3)}-${original.endTime.toFixed(3)}`;
      translation = translationMaps.exact.get(exactKey);
      
      // 2. 如果精确匹配失败，尝试索引匹配
      if (!translation) {
        translation = translationMaps.index.get(index);
      }
      
      // 3. 如果索引匹配失败，尝试模糊时间匹配
      if (!translation) {
        const fuzzyKey = `${Math.round(original.startTime * 10)}-${Math.round(original.endTime * 10)}`;
        translation = translationMaps.fuzzy.get(fuzzyKey);
      }
      
      mergedSubtitles.push({
        ...original,
        originalText: original.text, // 保存原始英文文本
        translation: translation || null, // 翻译文本
        hasTranslation: !!translation
      });
    });

    return mergedSubtitles;
  }

  /**
   * 验证SRT内容格式
   * @param {string} srtContent - SRT文件内容
   * @returns {Object} 验证结果
   */
  static validate(srtContent) {
    if (!srtContent || typeof srtContent !== 'string') {
      return {
        valid: false,
        error: 'SRT内容为空或格式无效'
      };
    }

    const blocks = srtContent
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n')
      .split(/\n\s*\n/)
      .filter(block => block.trim());

    if (blocks.length === 0) {
      return {
        valid: false,
        error: '没有找到有效的字幕段落'
      };
    }

    let validBlocks = 0;
    let errors = [];

    for (let i = 0; i < Math.min(blocks.length, 5); i++) { // 只检查前5个段落
      const lines = blocks[i].trim().split('\n');
      
      if (lines.length < 3) {
        errors.push(`段落 ${i + 1}: 行数不足`);
        continue;
      }

      // 检查序号
      const index = parseInt(lines[0].trim());
      if (isNaN(index)) {
        errors.push(`段落 ${i + 1}: 序号格式错误`);
        continue;
      }

      // 检查时间轴
      const timeMatch = lines[1].match(/(\d{2}:\d{2}:\d{2}[,.]?\d{0,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]?\d{0,3})/);
      if (!timeMatch) {
        errors.push(`段落 ${i + 1}: 时间轴格式错误`);
        continue;
      }

      validBlocks++;
    }

    const validRate = validBlocks / Math.min(blocks.length, 5);
    
    return {
      valid: validRate >= 0.8, // 80%以上的段落有效就认为格式正确
      error: errors.length > 0 ? errors[0] : null,
      totalBlocks: blocks.length,
      validBlocks: validBlocks,
      validRate: validRate
    };
  }

  /**
   * 获取字幕统计信息
   * @param {Array} subtitles - 字幕数组
   * @returns {Object} 统计信息
   */
  static getStats(subtitles) {
    if (!Array.isArray(subtitles) || subtitles.length === 0) {
      return {
        count: 0,
        duration: 0,
        avgDuration: 0,
        avgTextLength: 0
      };
    }

    const totalDuration = subtitles.reduce((sum, sub) => 
      sum + (sub.endTime - sub.startTime), 0
    );
    
    const totalTextLength = subtitles.reduce((sum, sub) => 
      sum + (sub.text ? sub.text.length : 0), 0
    );

    return {
      count: subtitles.length,
      duration: totalDuration,
      avgDuration: totalDuration / subtitles.length,
      avgTextLength: totalTextLength / subtitles.length,
      firstTime: subtitles[0].startTime,
      lastTime: subtitles[subtitles.length - 1].endTime
    };
  }
}

// 如果在Node.js环境中运行，导出模块
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SRTParser;
}