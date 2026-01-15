/**
 * 翻译模块 - 移植自 Python optimizer.py
 * 实现批次翻译、边界优化、上下文注入
 */

import { setupLogger } from '../utils/logger.js';
import { buildTranslatePrompt, buildSingleTranslatePrompt } from './prompts.js';
import { parseLlmResponse } from '../utils/json-repair.js';
import { getLanguageName } from '../utils/language.js';
import type { TranslatorConfig, SummaryResult, TranslatedEntry } from '../types/index.js';

const logger = setupLogger('translator');

/**
 * OpenAI API 客户端接口
 */
interface OpenAIClient {
  callChat(systemPrompt: string, userPrompt: string, options?: {
    temperature?: number;
    timeout?: number;
  }): Promise<string>;
}

/**
 * 检查句子是否完整
 */
function isSentenceComplete(text: string): boolean {
  const sentenceEndMarkers = ['.', '!', '?', '。', '！', '？', '…'];
  const badEndWords = ['and', 'or', 'but', 'so', 'yet', 'for', 'nor', 'in', 'on', 'at', 'to', 'with', 'by', 'as'];

  text = text.trim();
  if (!text) return true;

  // 检查最后一个字符是否是句子结束标志
  if (sentenceEndMarkers.some(marker => text.endsWith(marker))) {
    return true;
  }

  // 检查是否以不好的词结尾
  const lowerText = text.toLowerCase();
  for (const word of badEndWords) {
    if (lowerText.endsWith(' ' + word) || lowerText === word) {
      return false;
    }
  }

  // 如果句子太短，可能不完整
  const words = text.split(/\s+/);
  if (words.length < 3) {
    return false;
  }

  return true;
}

/**
 * 构建翻译参考信息
 */
function buildReferenceInfo(summary: SummaryResult): string {
  const parts: string[] = [];

  // 添加上下文信息
  if (summary.context) {
    parts.push(`Context: ${summary.context.type} - ${summary.context.topic}`);
  }

  // 添加纠错映射
  if (summary.corrections && Object.keys(summary.corrections).length > 0) {
    parts.push(`Apply corrections: ${JSON.stringify(summary.corrections)}`);
  }

  // 添加不翻译列表
  if (summary.do_not_translate && summary.do_not_translate.length > 0) {
    parts.push(`Keep in original: ${summary.do_not_translate.join(', ')}`);
  }

  // 添加规范术语
  if (summary.canonical_terms && summary.canonical_terms.length > 0) {
    const terms = summary.canonical_terms.slice(0, 10); // 限制显示前10个
    parts.push(`Use canonical forms: ${terms.join(', ')}`);
  }

  return parts.length > 0 ? '\n\n<reference>\n' + parts.join('\n') + '\n</reference>' : '';
}

/**
 * 翻译器类
 */
export class Translator {
  private client: OpenAIClient;
  private config: TranslatorConfig;
  private batchLogs: Array<{ type: string; id: number; original: string; optimized: string }> = [];

  constructor(client: OpenAIClient, config: TranslatorConfig) {
    this.client = client;
    this.config = config;
  }

  /**
   * 批量翻译字幕
   * @param subtitles 字幕数据 {index: text}
   * @param summary 内容总结结果
   * @param onProgress 进度回调
   */
  async translate(
    subtitles: Record<string, string>,
    summary: SummaryResult,
    onProgress?: (current: number, total: number) => void
  ): Promise<TranslatedEntry[]> {
    this.batchLogs = [];

    const targetLanguage = getLanguageName(this.config.targetLanguage);
    const batchSize = this.config.batchSize;

    // 构建批次，确保边界在完整句子处
    const items = Object.entries(subtitles);
    const batches = this.createBatches(items, batchSize);

    logger.info(`📋 翻译任务规划: ${batches.length}个批次，每批次约${batchSize}条字幕`);

    // 并发翻译所有批次（与 Python 版本一致）
    const tasks = batches.map((batch, i) =>
      this.translateBatch(
        batch,
        summary,
        targetLanguage,
        i + 1,
        batches.length
      ).catch(error => {
        logger.error(`❌ 批次 ${i + 1} 翻译失败: ${error}`);
        // 使用单条翻译降级处理
        return this.translateSingle(batch, targetLanguage);
      })
    );

    // 等待所有批次完成，并发执行
    logger.info(`⚡ 启动并发翻译: ${batches.length} 个批次同时处理`);
    const batchResults = await Promise.all(tasks);

    // 合并结果
    const results: TranslatedEntry[] = [];
    for (const batchResult of batchResults) {
      results.push(...batchResult);
    }

    // 打印批次日志汇总
    this.printBatchLogs();

    // 按 ID 排序
    results.sort((a, b) => a.index - b.index);

    // ============ 关键改进：二次失败检查和重试 ============
    // 模拟 Python 版本的 optimizer.py:94-112 行逻辑
    // 检查翻译结果，找出失败的条目
    const failedEntries = results.filter(r => r.translation.startsWith('[翻译失败]'));

    if (failedEntries.length > 0) {
      logger.info(`🔄 发现 ${failedEntries.length} 个字幕翻译失败，使用单条翻译再次尝试`);

      // 构建失败字幕映射
      const failedSubtitles: [string, string][] = failedEntries.map(entry => [
        String(entry.index),
        entry.original,
      ]);

      try {
        // 二次重试（使用单条翻译）
        const retryResults = await this.translateSingle(failedSubtitles, targetLanguage);

        // 更新成功的重试结果
        let successCount = 0;
        for (const retryResult of retryResults) {
          if (!retryResult.translation.startsWith('[翻译失败]')) {
            const idx = results.findIndex(r => r.index === retryResult.index);
            if (idx >= 0) {
              results[idx] = retryResult;
              successCount++;
              logger.info(`✅ 字幕 ID ${retryResult.index} 二次重试成功`);
            }
          }
        }

        logger.info(`📊 二次重试结果: ${successCount}/${failedEntries.length} 条字幕成功翻译`);

      } catch (error) {
        logger.error(`❌ 二次重试过程出错: ${error}`);
      }
    }
    // ============ 二次失败检查和重试结束 ============

    return results;
  }

  /**
   * 创建批次，优化边界
   */
  private createBatches(items: [string, string][], batchSize: number): [string, string][][] {
    const batches: [string, string][][] = [];
    let i = 0;
    let adjustedCount = 0;

    while (i < items.length) {
      let endIdx = Math.min(i + batchSize, items.length);

      // 如果不是最后一个批次，检查边界
      if (endIdx < items.length) {
        const lastText = items[endIdx - 1][1];

        if (!isSentenceComplete(lastText)) {
          // 向前查找完整句子
          let completeIdx = endIdx - 1;
          while (completeIdx > i && !isSentenceComplete(items[completeIdx - 1][1])) {
            completeIdx--;
          }

          if (completeIdx > i) {
            logger.info(`调整批次边界: ${endIdx} -> ${completeIdx} (确保句子完整性)`);
            endIdx = completeIdx;
            adjustedCount++;
          }
        }
      }

      batches.push(items.slice(i, endIdx));
      i = endIdx;
    }

    if (adjustedCount > 0) {
      logger.info(`🔧 已优化${adjustedCount}个批次边界，确保句子完整性`);
    }

    return batches;
  }

  /**
   * 翻译单个批次
   * 注意：重试逻辑已移至 OpenAIClient，此处不再重复
   */
  private async translateBatch(
    batch: [string, string][],
    summary: SummaryResult,
    targetLanguage: string,
    batchNum: number,
    totalBatches: number
  ): Promise<TranslatedEntry[]> {
    const batchInfo = `[批次${batchNum}/${totalBatches}]`;
    logger.info(`🌍 ${batchInfo} 翻译 ${batch.length} 条字幕`);

    // 构建输入
    const inputObj: Record<string, string> = Object.fromEntries(batch);

    // 构建 Prompt
    const systemPrompt = buildTranslatePrompt({ targetLanguage });
    const referenceInfo = buildReferenceInfo(summary);
    const userPrompt = `Correct and translate the following subtitles into ${targetLanguage}:
<subtitles>${JSON.stringify(inputObj, null, 2)}</subtitles>${referenceInfo}`;

    logger.info(`📤 ${batchInfo} 提交给LLM的字幕数据 (共${batch.length}条):`);
    logger.info(`   输入JSON: ${JSON.stringify(inputObj)}`);

    // 调用 API（OpenAIClient 已内置重试）
    const response = await this.client.callChat(systemPrompt, userPrompt, {
      temperature: 0.7,
      timeout: 80000,
    });

    logger.info(`📥 ${batchInfo} LLM原始返回数据:\n${response}`);

    // 解析响应
    const responseContent = this.normalizeResponse(parseLlmResponse(response), batchInfo);

    // 构建结果
    return batch.map(([key, originalText]) => {
      const entry = responseContent[key];
      const optimized = entry?.optimized_subtitle || originalText;
      const translation = entry?.translation || `[翻译失败] ${originalText}`;

      if (!entry) {
        logger.warn(`⚠️ API返回结果缺少字幕ID: ${key}`);
      }

      // 记录优化日志
      if (originalText !== optimized) {
        this.batchLogs.push({
          type: 'content_optimization',
          id: parseInt(key, 10),
          original: originalText,
          optimized,
        });
      }

      return {
        index: parseInt(key, 10),
        startTime: 0,
        endTime: 0,
        original: originalText,
        optimized,
        translation,
      };
    });
  }

  /**
   * 标准化 LLM 响应格式
   * 将数组格式转换为对象格式
   */
  private normalizeResponse(
    content: unknown,
    batchInfo: string
  ): Record<string, { optimized_subtitle?: string; translation?: string }> {
    if (!Array.isArray(content)) {
      return (content as Record<string, { optimized_subtitle?: string; translation?: string }>) || {};
    }

    logger.warn(`⚠️ ${batchInfo} LLM返回了array而非object，尝试转换`);
    const result: Record<string, { optimized_subtitle: string; translation: string }> = {};

    for (const item of content) {
      if (typeof item !== 'object' || item === null) continue;

      const record = item as Record<string, unknown>;
      const itemId = record.id || record.subtitle_id || record.key;
      if (!itemId) continue;

      result[String(itemId)] = {
        optimized_subtitle: String(record.optimized_subtitle || record.optimized || ''),
        translation: String(record.translation || ''),
      };
    }

    if (Object.keys(result).length > 0) {
      logger.info(`✅ ${batchInfo} 成功转换array为object，包含${Object.keys(result).length}个条目`);
    }

    return result;
  }

  /**
   * 单条翻译（降级处理）
   * 注意：重试逻辑已移至 OpenAIClient，此处不再重复
   */
  private async translateSingle(
    batch: [string, string][],
    targetLanguage: string
  ): Promise<TranslatedEntry[]> {
    logger.info(`[+]正在单条翻译字幕，共${batch.length}条`);

    const systemPrompt = buildSingleTranslatePrompt({ targetLanguage });
    const results: TranslatedEntry[] = [];

    for (const [key, value] of batch) {
      let translation: string;

      try {
        logger.info(`[+]正在翻译字幕ID: ${key}`);

        const response = await this.client.callChat(systemPrompt, value, {
          temperature: 0.7,
          timeout: 80000,
        });

        translation = response.trim();
        logger.info(`单条翻译原文: ${value}`);
        logger.info(`单条翻译结果: ${translation}`);
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        logger.error(`❌ 字幕 ID ${key} 单条翻译失败: ${errorMsg}`);
        translation = `[翻译失败] ${value}`;
      }

      results.push({
        index: parseInt(key, 10),
        startTime: 0,
        endTime: 0,
        original: value,
        optimized: value,
        translation,
      });
    }

    return results;
  }

  /**
   * 打印批次日志汇总
   */
  private printBatchLogs(): void {
    if (this.batchLogs.length === 0) return;

    logger.info('📊 字幕优化结果汇总');

    let formatChanges = 0;
    let contentChanges = 0;

    for (const log of this.batchLogs) {
      if (log.type === 'content_optimization') {
        logger.info(`🔧 字幕ID ${log.id} - 内容优化:`);
        logger.info(`   原文: ${log.original}`);
        logger.info(`   优化: ${log.optimized}`);

        // 简单判断是否只有格式变化
        const normalizedOriginal = log.original.toLowerCase().replace(/[^\w\s]/g, '');
        const normalizedOptimized = log.optimized.toLowerCase().replace(/[^\w\s]/g, '');

        if (normalizedOriginal === normalizedOptimized) {
          formatChanges++;
        } else {
          contentChanges++;
        }
      }
    }

    logger.info('📈 优化统计:');
    logger.info(`   格式优化: ${formatChanges} 项`);
    logger.info(`   内容修改: ${contentChanges} 项`);
    logger.info(`   总计修改: ${formatChanges + contentChanges} 项`);
    logger.info('✅ 字幕优化汇总完成');
  }
}

/**
 * 创建翻译器实例
 */
export function createTranslator(client: OpenAIClient, config: TranslatorConfig): Translator {
  return new Translator(client, config);
}
