/**
 * Chrome Extension 适配层
 * 保持与现有 translator.js 相同的全局接口
 */

import { TranslatorService, createTranslatorService } from '../services/translator-service.js';
import { loadConfig } from '../utils/config.js';
import { getLanguageName, LANGUAGE_MAPPING } from '../utils/language.js';
import type { TranslatorConfig, SubtitleEntry, BilingualSubtitles } from '../types/index.js';

/**
 * 翻译服务包装器 - 保持与现有接口兼容
 */
class TranslatorServiceWrapper {
  private service: TranslatorService | null = null;
  private config: TranslatorConfig | null = null;
  public isTranslating = false;

  /**
   * 加载 API 配置
   */
  async loadConfig(): Promise<TranslatorConfig> {
    this.config = await loadConfig();
    this.service = createTranslatorService(this.config);
    return this.config;
  }

  /**
   * 获取目标语言名称
   */
  getTargetLanguageName(langCode: string): string {
    return getLanguageName(langCode);
  }

  /**
   * 调用 OpenAI API（兼容接口）
   */
  async callOpenAI(systemPrompt: string, userPrompt: string): Promise<string> {
    if (!this.config) {
      await this.loadConfig();
    }

    const response = await fetch(`${this.config!.openaiBaseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.config!.openaiApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.config!.translationModel,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.3,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({})) as { error?: { message?: string } };
      throw new Error(error.error?.message || `API请求失败: ${response.status}`);
    }

    const data = await response.json() as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    return data.choices?.[0]?.message?.content || '';
  }

  /**
   * 解析 JSON 响应
   */
  parseJsonResponse(content: string): unknown {
    try {
      return JSON.parse(content);
    } catch {
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          return JSON.parse(jsonMatch[0]);
        } catch {
          console.error('JSON解析失败:', content);
          return null;
        }
      }
      return null;
    }
  }

  /**
   * 执行完整翻译流程
   * @param subtitles 原始字幕数组（时间戳单位：秒）
   * @param targetLang 目标语言代码
   * @param onProgress 进度回调
   */
  async translateFull(
    subtitles: Array<{ startTime: number; endTime: number; text: string }>,
    targetLang = 'zh',
    onProgress: ((step: string, current: number, total: number) => void) | null = null
  ): Promise<BilingualSubtitles> {
    if (this.isTranslating) {
      throw new Error('翻译正在进行中');
    }

    this.isTranslating = true;

    // 保存翻译状态到 storage
    const saveProgress = async (step: string, current: number, total: number): Promise<void> => {
      if (typeof chrome !== 'undefined' && chrome.storage) {
        await chrome.storage.local.set({
          translationProgress: {
            isTranslating: true,
            step,
            current,
            total,
            timestamp: Date.now(),
          },
        });
      }
      if (onProgress) onProgress(step, current, total);
    };

    try {
      // 加载配置
      if (!this.config) {
        await this.loadConfig();
      }

      // 更新目标语言
      this.config!.targetLanguage = targetLang;
      this.service = createTranslatorService(this.config!);

      // 转换字幕格式，同时将秒转换为毫秒（与 CLI 版本和核心服务保持一致）
      // 扩展从 YouTube 获取的字幕时间是秒，但核心翻译服务期望毫秒
      const entries: SubtitleEntry[] = subtitles.map((sub, idx) => ({
        index: idx + 1,
        startTime: Math.round(sub.startTime * 1000),  // 秒 -> 毫秒
        endTime: Math.round(sub.endTime * 1000),      // 秒 -> 毫秒
        text: sub.text,
      }));

      // 执行翻译
      const result = await this.service!.translateFull(entries, {
        onProgress: async (step, current, total) => {
          await saveProgress(step, current, total);
        },
      });

      await saveProgress('complete', 3, 3);

      // 将翻译结果的时间戳从毫秒转换回秒（用于与视频 currentTime 比较）
      const convertedResult: BilingualSubtitles = {
        english: result.english.map(entry => ({
          ...entry,
          startTime: entry.startTime / 1000,  // 毫秒 -> 秒
          endTime: entry.endTime / 1000,      // 毫秒 -> 秒
        })),
        chinese: result.chinese.map(entry => ({
          ...entry,
          startTime: entry.startTime / 1000,  // 毫秒 -> 秒
          endTime: entry.endTime / 1000,      // 毫秒 -> 秒
        })),
      };

      return convertedResult;

    } finally {
      this.isTranslating = false;
      // 清除翻译进度状态
      if (typeof chrome !== 'undefined' && chrome.storage) {
        chrome.storage.local.remove('translationProgress');
      }
    }
  }

  /**
   * 取消翻译
   */
  cancelTranslation(): void {
    this.isTranslating = false;
    if (this.service) {
      this.service.cancel();
    }
  }
}

// 创建全局实例
const translatorService = new TranslatorServiceWrapper();

// 导出到全局（支持浏览器和 Service Worker）
interface GlobalExports {
  TranslatorService: typeof TranslatorServiceWrapper;
  translatorService: TranslatorServiceWrapper;
  LANGUAGE_MAPPING: typeof LANGUAGE_MAPPING;
}

const globalExports = globalThis as unknown as GlobalExports;
globalExports.TranslatorService = TranslatorServiceWrapper;
globalExports.translatorService = translatorService;
globalExports.LANGUAGE_MAPPING = LANGUAGE_MAPPING;

// 导出模块
export { TranslatorServiceWrapper as TranslatorService, translatorService, LANGUAGE_MAPPING };
