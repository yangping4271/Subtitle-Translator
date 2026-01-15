/**
 * OpenAI API 客户端
 * 支持 Node.js 和浏览器环境
 */

import { setupLogger } from '../utils/logger.js';
import { withRetry } from '../utils/retry.js';
import { TranslationError } from '../utils/error-handler.js';
import type { TranslatorConfig } from '../types/index.js';

const logger = setupLogger('openai-client');

// 检测运行环境
const isNode = typeof process !== 'undefined' && process.versions?.node;

/**
 * OpenAI API 客户端
 */
export class OpenAIClient {
  private baseUrl: string;
  private apiKey: string;
  private model: string;

  constructor(config: TranslatorConfig, modelType: 'split' | 'summary' | 'translation' = 'translation') {
    this.baseUrl = config.openaiBaseUrl;
    this.apiKey = config.openaiApiKey;

    // 根据类型选择模型
    switch (modelType) {
      case 'split':
        this.model = config.splitModel;
        break;
      case 'summary':
        this.model = config.summaryModel;
        break;
      case 'translation':
      default:
        this.model = config.translationModel;
        break;
    }
  }

  /**
   * 调用 Chat API（带自动重试）
   */
  async callChat(
    systemPrompt: string,
    userPrompt: string,
    options: { temperature?: number; timeout?: number } = {}
  ): Promise<string> {
    const { temperature = 0.7, timeout = 80000 } = options;

    if (!this.apiKey) {
      throw TranslationError.fromError(
        new Error('API 密钥未配置'),
        'OpenAI Client'
      );
    }

    const url = `${this.baseUrl}/chat/completions`;
    const body = {
      model: this.model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      temperature,
    };

    // 使用 withRetry 包装 API 调用，自动重试 2 次
    return withRetry(
      async () => {
        try {
          let response: Response;

          if (isNode) {
            // Node.js 环境
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeout);

            response = await fetch(url, {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(body),
              signal: controller.signal,
            });

            clearTimeout(timeoutId);
          } else {
            // 浏览器环境
            response = await fetch(url, {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(body),
            });
          }

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = (errorData as { error?: { message?: string } })?.error?.message ||
              `API 请求失败: ${response.status}`;
            throw TranslationError.fromError(new Error(errorMessage), 'API Request');
          }

          const data = await response.json() as {
            choices?: Array<{ message?: { content?: string } }>;
          };

          const content = data.choices?.[0]?.message?.content;

          if (!content) {
            throw new Error('API 返回内容为空');
          }

          return content;

        } catch (error) {
          if (error instanceof Error) {
            if (error.name === 'AbortError') {
              throw new Error('请求超时');
            }
            throw error;
          }
          throw new Error(`API 调用失败: ${error}`);
        }
      },
      {
        maxRetries: 2,
        delays: [1000, 2000],
        operationName: `OpenAI API (${this.model})`,
      }
    );
  }
}

/**
 * 创建 OpenAI 客户端
 */
export function createOpenAIClient(
  config: TranslatorConfig,
  modelType: 'split' | 'summary' | 'translation' = 'translation'
): OpenAIClient {
  return new OpenAIClient(config, modelType);
}
