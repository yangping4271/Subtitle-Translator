/**
 * 总结模块 - 移植自 Python summarizer.py
 * 分析字幕内容，提取翻译所需的上下文信息
 */

import { setupLogger } from '../utils/logger.js';
import { SUMMARIZER_PROMPT } from './prompts.js';
import { parseLlmResponse } from '../utils/json-repair.js';
import type { SummaryResult, TranslatorConfig } from '../types/index.js';

const logger = setupLogger('summarizer');

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
 * 从文件路径提取可读性信息
 * @param inputFile 输入文件路径
 */
export function extractFileContext(inputFile: string): {
  readableName: string;
  folderPath: string;
} {
  // 简化实现：提取文件名和路径
  const parts = inputFile.replace(/\\/g, '/').split('/');
  const fileName = parts.pop() || '';

  // 获取不带扩展名的文件名
  const readableName = fileName
    .replace(/\.[^/.]+$/, '') // 移除扩展名
    .replace(/[_-]/g, ' ');   // 替换下划线和连字符为空格

  // 获取文件夹路径（最多3级）
  const parentParts = parts.slice(-3).map(p => p.replace(/[_-]/g, ' '));
  const folderPath = parentParts.join(' / ');

  return { readableName, folderPath };
}

/**
 * 构建上下文信息字符串
 */
function buildContextInfo(inputFile?: string, videoTitle?: string): string {
  const contextParts: string[] = [];

  if (inputFile) {
    const { readableName, folderPath } = extractFileContext(inputFile);
    if (folderPath) {
      contextParts.push(`Folder path: ${folderPath}`);
    }
    contextParts.push(`Filename: ${readableName}`);
  }

  if (videoTitle) {
    contextParts.push(`Video title: ${videoTitle}`);
  }

  return contextParts.join('\n');
}

/**
 * 总结器类
 */
export class Summarizer {
  private client: OpenAIClient;
  private config: TranslatorConfig;

  constructor(client: OpenAIClient, config: TranslatorConfig) {
    this.client = client;
    this.config = config;
  }

  /**
   * 总结字幕内容
   * @param subtitleContent 字幕内容文本
   * @param options 选项
   */
  async summarize(
    subtitleContent: string,
    options: { inputFile?: string; videoTitle?: string } = {}
  ): Promise<SummaryResult> {
    const { inputFile, videoTitle } = options;

    // 提取文件上下文信息
    const contextInfo = buildContextInfo(inputFile, videoTitle);

    if (inputFile) {
      const { readableName, folderPath } = extractFileContext(inputFile);
      logger.info(`📋 可读性文件名: ${readableName}`);
      if (folderPath) {
        logger.info(`📂 文件夹路径: ${folderPath}`);
      }
    }

    // 获取当前日期
    const currentDate = new Date().toISOString().split('T')[0];

    // 替换 prompt 中的日期占位符
    const promptWithDate = SUMMARIZER_PROMPT.replace('{current_date}', currentDate);

    // 构建系统提示
    const systemPrompt = `You are a precise subtitle summarizer. When processing proper nouns and product names:
1. Use BOTH the folder path AND filename as authoritative references for product names
2. Folder names often contain the correct product/topic names
3. Only correct terms that appear to be ASR errors based on:
   - Similar pronunciation
   - Context indicating they refer to the same thing
   - Mismatch with folder/filename context
4. Do not modify other technical terms or module names that are clearly different
${promptWithDate}`;

    // 构建用户提示
    const userPrompt = contextInfo
      ? `${contextInfo}\n\nContent:\n${subtitleContent}`
      : `Content:\n${subtitleContent}`;

    try {
      const response = await this.client.callChat(systemPrompt, userPrompt, {
        temperature: 0.7,
        timeout: 80000,
      });

      if (!response) {
        throw new Error('API 返回为空');
      }

      // 移除 <think></think> 标签
      const cleanedResponse = response.replace(/<think>[\s\S]*?<\/think>/g, '');

      // 解析 JSON 响应
      const result = parseLlmResponse(cleanedResponse) as unknown as SummaryResult;

      // 验证并填充缺失字段
      const summary: SummaryResult = {
        context: result.context || {
          type: 'unknown',
          topic: 'Unknown topic',
          formality: 'neutral',
        },
        corrections: result.corrections || {},
        canonical_terms: result.canonical_terms || [],
        do_not_translate: result.do_not_translate || [],
        style_guide: result.style_guide || {
          audience: 'general',
          technical_level: 'intermediate',
          tone: 'neutral',
        },
      };

      // 输出完整的总结内容（与 Python 版本一致）
      logger.info(`总结字幕内容:\n${JSON.stringify(summary, null, 2)}\n`);

      return summary;

    } catch (error) {
      logger.error(`内容分析失败: ${error}`);

      // 返回默认值
      return {
        context: {
          type: 'unknown',
          topic: 'Unknown',
          formality: 'neutral',
        },
        corrections: {},
        canonical_terms: [],
        do_not_translate: [],
        style_guide: {
          audience: 'general',
          technical_level: 'intermediate',
          tone: 'neutral',
        },
      };
    }
  }
}

/**
 * 创建总结器实例
 */
export function createSummarizer(client: OpenAIClient, config: TranslatorConfig): Summarizer {
  return new Summarizer(client, config);
}
