/**
 * 翻译服务 - 整合所有模块
 * 提供完整的翻译流程：断句 → 总结 → 翻译
 * 与 Python 版本 (service.py) 保持完全一致的逻辑
 */

import { setupLogger } from '../utils/logger.js';
import { createOpenAIClient } from './openai-client.js';
import { mergeSegmentsBatch, countWords } from '../core/splitter.js';
import { SubtitleData } from '../core/subtitle-data.js';
import { createSummarizer } from '../core/summarizer.js';
import { createTranslator } from '../core/translator.js';
import type {
  TranslatorConfig,
  SubtitleEntry,
  TranslatedEntry,
  BilingualSubtitles,
  ProgressCallback,
  TranslateOptions,
  SummaryResult,
} from '../types/index.js';

const logger = setupLogger('translator-service');

/**
 * 翻译服务类
 */
export class TranslatorService {
  private config: TranslatorConfig;
  private isTranslating = false;

  constructor(config: TranslatorConfig) {
    this.config = config;
  }

  /**
   * 执行完整翻译流程
   * 参考 Python 版本: service.py:translate_srt
   *
   * 流程：
   * 1. 检测字幕类型（单词级 vs 片段级）
   * 2. 如果是片段级，转换为单词级（音素理论）
   * 3. 并行预处理：断句 + 内容总结
   * 4. 翻译
   * 5. 对齐时间戳
   *
   * @param subtitles 原始字幕数组
   * @param options 翻译选项
   */
  async translateFull(
    subtitles: SubtitleEntry[],
    options: TranslateOptions = {}
  ): Promise<BilingualSubtitles> {
    if (this.isTranslating) {
      throw new Error('翻译正在进行中');
    }

    this.isTranslating = true;
    const { inputFile, videoTitle, onProgress } = options;

    try {
      // 创建字幕数据对象
      const subtitleData = new SubtitleData(subtitles);
      logger.info(`📊 字幕统计: 共 ${subtitleData.length()} 条字幕`);
      logger.info(`字幕内容预览: ${subtitleData.toText().slice(0, 100)}...`);

      // 检查字幕是否为空
      if (subtitleData.length() === 0) {
        throw new Error('SRT文件为空，无法进行翻译');
      }

      // 并行预处理阶段（与 Python 版本一致）
      logger.info('\n⚡ 并行预处理阶段 开始');

      // 准备原始字幕内容用于总结
      const originalSubtitleContent = subtitleData.toText();

      // 启动断句任务
      const executeSplitting = async (data: SubtitleData): Promise<SubtitleData> => {
        logger.info('\n✂️ 字幕断句处理 开始');

        // 保存原始字幕数据（用于后续时间戳匹配）
        const originalData = data;

        // 🔍 调试：打印原始数据信息
        const originalSegments = originalData.getSegments();
        logger.info(`🔍 原始数据: ${originalSegments.length} 条字幕`);
        if (originalSegments.length > 0) {
          logger.info(`🔍 原始时间戳: ${originalSegments[0].startTime}s - ${originalSegments[originalSegments.length - 1].endTime}s`);
          logger.info(`🔍 第一条: "${originalSegments[0].text}"`);
          logger.info(`🔍 第一条时长: ${originalSegments[0].endTime - originalSegments[0].startTime}s`);
        }

        // 检查字幕类型并统一转换为单词级别（核心创新功能）
        let processData = data;

        if (data.isWordTimestamp()) {
          logger.info('检测到单词级别时间戳，执行合并断句');
        } else {
          logger.info('检测到片段级别时间戳，先转换为单词级别');
          // 统一转换为单词级别字幕
          // 使用音素级时间戳分配，支持多语言处理
          processData = data.splitToWordSegments();
          logger.info(`转换完成，生成 ${processData.length()} 个单词级别片段`);

          // 🔍 调试：打印转换后的数据信息
          const processSegments = processData.getSegments();
          if (processSegments.length > 0) {
            logger.info(`🔍 转换后时间戳: ${processSegments[0].startTime}s - ${processSegments[processSegments.length - 1].endTime}s`);
            logger.info(`🔍 转换后第一条: "${processSegments[0].text}"`);
            logger.info(`🔍 转换后第一条时长: ${processSegments[0].endTime - processSegments[0].startTime}s`);
          }
        }

        // 执行统一的断句处理流程
        logger.info(`🤖 使用模型: ${this.config.splitModel}`);
        logger.info(`📏 句子长度限制: ${this.config.maxWordCountEnglish} 字`);

        const splitClient = createOpenAIClient(this.config, 'split');
        // 传入原始数据和处理数据
        const result = await mergeSegmentsBatch(processData, originalData, splitClient, this.config, 3);

        logger.info(`✅ 字幕断句处理 完成\n`);
        logger.info(`✅ 断句完成 (优化为 ${result.length()} 句)`);

        return result;
      };

      // 启动总结任务
      const executeSummarization = async (content: string, file?: string): Promise<SummaryResult> => {
        const summaryClient = createOpenAIClient(this.config, 'summary');
        const summarizer = createSummarizer(summaryClient, this.config);

        logger.info(`🤖 使用模型: ${this.config.summaryModel}`);

        // 使用前20句进行总结（与原始内容而非断句后的内容）
        const sampleText = content.split(/[.!?]+/).slice(0, 20).join('. ');
        const summary = await summarizer.summarize(sampleText, { inputFile: file, videoTitle });

        return summary;
      };

      // 并行执行断句和总结任务
      const [splitResult, summaryResult] = await Promise.all([
        executeSplitting(subtitleData),
        executeSummarization(originalSubtitleContent, inputFile),
      ]);

      logger.info('⚡ 并行预处理阶段 完成\n');

      if (onProgress) onProgress('split', 1, 3);
      if (onProgress) onProgress('summary', 2, 3);

      // 构建优化后的字幕索引
      const optimizedSubtitles: Record<string, string> = {};
      const splitSegments = splitResult.getSegments();

      splitSegments.forEach((seg, idx) => {
        optimizedSubtitles[String(idx + 1)] = seg.text;
      });

      // 步骤3：翻译
      logger.info('🌐 步骤3: 翻译字幕...');
      if (onProgress) onProgress('translate', 2, 3);

      const translationClient = createOpenAIClient(this.config, 'translation');
      const translator = createTranslator(translationClient, this.config);

      const translatedEntries = await translator.translate(
        optimizedSubtitles,
        summaryResult,
        (current, total) => {
          if (onProgress) {
            const progress = 2 + (current / total);
            onProgress('translate', progress, 3);
          }
        }
      );

      if (onProgress) onProgress('complete', 3, 3);

      // 构建双语字幕结果
      const result = this.buildBilingualResult(splitSegments, translatedEntries);

      logger.info(`✅ 翻译完成: ${result.english.length} 条双语字幕`);

      return result;

    } finally {
      this.isTranslating = false;
    }
  }

  /**
   * 构建双语字幕结果
   * 使用断句后的时间戳信息
   */
  private buildBilingualResult(
    splitSegments: SubtitleEntry[],
    translatedEntries: TranslatedEntry[]
  ): BilingualSubtitles {
    const english: SubtitleEntry[] = [];
    const chinese: SubtitleEntry[] = [];

    for (let i = 0; i < translatedEntries.length && i < splitSegments.length; i++) {
      const entry = translatedEntries[i];
      const segment = splitSegments[i];

      english.push({
        index: i + 1,
        startTime: segment.startTime,
        endTime: segment.endTime,
        text: entry.optimized,
      });

      chinese.push({
        index: i + 1,
        startTime: segment.startTime,
        endTime: segment.endTime,
        text: entry.translation,
      });
    }

    return { english, chinese };
  }

  /**
   * 取消翻译
   */
  cancel(): void {
    this.isTranslating = false;
  }

  /**
   * 检查是否正在翻译
   */
  get translating(): boolean {
    return this.isTranslating;
  }
}

/**
 * 创建翻译服务实例
 */
export function createTranslatorService(config: TranslatorConfig): TranslatorService {
  return new TranslatorService(config);
}
