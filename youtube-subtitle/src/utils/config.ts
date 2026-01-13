/**
 * 配置管理 - 支持 Node.js 和浏览器环境
 */

import type { TranslatorConfig, ApiConfig } from '../types/index.js';

// 检测运行环境
const isNode = typeof process !== 'undefined' && process.versions?.node;

// 默认配置
const DEFAULT_CONFIG: TranslatorConfig = {
  openaiBaseUrl: 'https://api.openai.com/v1',
  openaiApiKey: '',
  splitModel: 'gpt-4o-mini',
  summaryModel: 'gpt-4o-mini',
  translationModel: 'gpt-4o',
  targetLanguage: 'zh',
  maxWordCountEnglish: 19,
  threadNum: 18,
  batchSize: 20, // 与 Python 版本一致
  // 阈值倍数（与 Python 版本一致）
  toleranceMultiplier: 1.2,
  warningMultiplier: 1.5,
  maxMultiplier: 2.0,
};

/**
 * 从环境变量加载配置（Node.js 环境）
 * 注意：此函数仅在 CLI 构建中使用，浏览器构建会被 tree-shaking 移除
 */
async function loadConfigFromEnv(): Promise<TranslatorConfig> {
  // 在浏览器环境下直接返回默认配置
  if (!isNode) {
    return { ...DEFAULT_CONFIG };
  }

  // Node.js 环境：从环境变量读取
  // 注意：dotenv 配置应该在 CLI 入口点完成
  return {
    openaiBaseUrl: process.env.OPENAI_BASE_URL || DEFAULT_CONFIG.openaiBaseUrl,
    openaiApiKey: process.env.OPENAI_API_KEY || '',
    splitModel: process.env.SPLIT_MODEL || DEFAULT_CONFIG.splitModel,
    summaryModel: process.env.SUMMARY_MODEL || DEFAULT_CONFIG.summaryModel,
    translationModel: process.env.TRANSLATION_MODEL || DEFAULT_CONFIG.translationModel,
    targetLanguage: process.env.TARGET_LANGUAGE || DEFAULT_CONFIG.targetLanguage,
    maxWordCountEnglish: parseInt(process.env.MAX_WORD_COUNT || '19', 10),
    threadNum: parseInt(process.env.THREAD_NUM || '18', 10),
    batchSize: parseInt(process.env.BATCH_SIZE || '20', 10),
    toleranceMultiplier: parseFloat(process.env.TOLERANCE_MULTIPLIER || '1.2'),
    warningMultiplier: parseFloat(process.env.WARNING_MULTIPLIER || '1.5'),
    maxMultiplier: parseFloat(process.env.MAX_MULTIPLIER || '2.0'),
  };
}

/**
 * 从 Chrome Storage 加载配置（浏览器环境）
 */
async function loadConfigFromStorage(): Promise<TranslatorConfig> {
  return new Promise((resolve) => {
    if (typeof chrome !== 'undefined' && chrome.storage) {
      chrome.storage.local.get(['apiConfig'], (result) => {
        const apiConfig: ApiConfig = result.apiConfig || {};
        resolve({
          ...DEFAULT_CONFIG,
          openaiBaseUrl: apiConfig.openaiBaseUrl || DEFAULT_CONFIG.openaiBaseUrl,
          openaiApiKey: apiConfig.openaiApiKey || '',
          splitModel: apiConfig.llmModel || DEFAULT_CONFIG.splitModel,
          summaryModel: apiConfig.llmModel || DEFAULT_CONFIG.summaryModel,
          translationModel: apiConfig.llmModel || DEFAULT_CONFIG.translationModel,
          targetLanguage: apiConfig.targetLanguage || DEFAULT_CONFIG.targetLanguage,
        });
      });
    } else {
      resolve(DEFAULT_CONFIG);
    }
  });
}

/**
 * 加载配置 - 自动检测环境
 */
export async function loadConfig(): Promise<TranslatorConfig> {
  if (isNode) {
    return loadConfigFromEnv();
  } else {
    return loadConfigFromStorage();
  }
}

/**
 * 获取默认配置
 */
export function getDefaultConfig(): TranslatorConfig {
  return { ...DEFAULT_CONFIG };
}

/**
 * 验证配置
 */
export function validateConfig(config: TranslatorConfig): string[] {
  const errors: string[] = [];

  if (!config.openaiApiKey) {
    errors.push('API 密钥未配置');
  }

  if (!config.openaiBaseUrl) {
    errors.push('API 地址未配置');
  }

  if (config.maxWordCountEnglish < 5 || config.maxWordCountEnglish > 50) {
    errors.push('最大单词数应在 5-50 之间');
  }

  if (config.batchSize < 10 || config.batchSize > 100) {
    errors.push('批次大小应在 10-100 之间（推荐: 20）');
  }

  return errors;
}
