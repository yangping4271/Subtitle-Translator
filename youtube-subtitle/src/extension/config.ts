/**
 * YouTube Subtitle Translator - 统一配置中心
 * ============================================
 * 所有默认值的单一数据源 (Single Source of Truth)
 */

import type {
  SubtitleStyleSettings,
  SubtitleConfig,
  ModelOption,
  LanguageOption,
  ApiConfig,
} from '../types';

/** 默认 API 配置 */
export const DEFAULT_API_CONFIG: ApiConfig = {
  openaiBaseUrl: 'https://api.openai.com/v1',
  openaiApiKey: '',
  llmModel: '',
  targetLanguage: 'zh',
};

/** 支持的 LLM 模型列表 */
export const SUPPORTED_MODELS: ModelOption[] = [
  { value: 'gpt-4o-mini', text: 'GPT-4o Mini (推荐)' },
  { value: 'gpt-4o', text: 'GPT-4o' },
  { value: 'gpt-4-turbo', text: 'GPT-4 Turbo' },
  { value: 'gpt-3.5-turbo', text: 'GPT-3.5 Turbo' },
  { value: 'claude-3-haiku-20240307', text: 'Claude 3 Haiku' },
  { value: 'claude-3-sonnet-20240229', text: 'Claude 3 Sonnet' },
  { value: 'google/gemini-3-flash-preview', text: 'Gemini 3 Flash (Preview)' },
  { value: 'google/gemini-flash-1.5', text: 'Gemini 1.5 Flash' },
  { value: 'google/gemini-pro-1.5', text: 'Gemini 1.5 Pro' },
  { value: 'custom', text: '自定义模型...' },
];

/** 支持的目标语言列表 */
export const SUPPORTED_LANGUAGES: LanguageOption[] = [
  { value: 'zh', text: '简体中文' },
  { value: 'zh-tw', text: '繁体中文' },
  { value: 'ja', text: '日文' },
  { value: 'ko', text: '韩文' },
  { value: 'en', text: 'English' },
  { value: 'fr', text: '法文' },
  { value: 'de', text: '德文' },
  { value: 'es', text: '西班牙文' },
];

/** 默认字幕配置 */
export const DEFAULT_SUBTITLE_CONFIG: SubtitleConfig = {
  english: {
    fontSize: 30,
    fontColor: '#FFFF00',
    fontFamily: '"Noto Serif", Georgia, serif',
    fontWeight: '700',
    textStroke: '2px #000000',
    textShadow: 'none',
    lineHeight: 1.3,
  },
  chinese: {
    fontSize: 28,
    fontColor: '#00FF00',
    fontFamily: '"Songti SC", serif',
    fontWeight: '900',
    textStroke: '2px #000000',
    textShadow: 'none',
    lineHeight: 1.4,
  },
  dpr: {
    enabled: true,
    compensationFactor: 0.4,
  },
  ui: {
    fontSizeMin: 16,
    fontSizeMax: 48,
  },
};

/** 获取默认 API 配置的深拷贝 */
export function getDefaultApiConfig(): ApiConfig {
  return JSON.parse(JSON.stringify(DEFAULT_API_CONFIG));
}

/** 获取英文字幕默认配置的深拷贝 */
export function getDefaultEnglishSettings(): SubtitleStyleSettings {
  return JSON.parse(JSON.stringify(DEFAULT_SUBTITLE_CONFIG.english));
}

/** 获取中文字幕默认配置的深拷贝 */
export function getDefaultChineseSettings(): SubtitleStyleSettings {
  return JSON.parse(JSON.stringify(DEFAULT_SUBTITLE_CONFIG.chinese));
}

/** 获取完整的默认配置（只读） */
export function getDefaultConfig(): SubtitleConfig {
  return DEFAULT_SUBTITLE_CONFIG;
}

/** 验证和补全设置对象 */
export function validateSettings(
  settings: Partial<SubtitleStyleSettings> | null | undefined,
  type: 'english' | 'chinese'
): SubtitleStyleSettings {
  const defaults =
    type === 'english' ? getDefaultEnglishSettings() : getDefaultChineseSettings();

  const validated: SubtitleStyleSettings = { ...defaults };

  if (settings) {
    for (const key of Object.keys(defaults) as Array<keyof SubtitleStyleSettings>) {
      if (settings[key] !== undefined) {
        (validated as Record<string, unknown>)[key] = settings[key];
      }
    }
  }

  return validated;
}

/** 检查设置对象是否为空 */
export function isEmptySettings(obj: unknown): boolean {
  return !obj || (typeof obj === 'object' && Object.keys(obj).length === 0);
}

// 浏览器环境：挂载到全局
declare global {
  interface Window {
    SubtitleConfig: {
      getDefaultEnglishSettings: typeof getDefaultEnglishSettings;
      getDefaultChineseSettings: typeof getDefaultChineseSettings;
      getDefaultConfig: typeof getDefaultConfig;
      validateSettings: typeof validateSettings;
      isEmptySettings: typeof isEmptySettings;
      DEFAULT_API_CONFIG: typeof DEFAULT_API_CONFIG;
      SUPPORTED_MODELS: typeof SUPPORTED_MODELS;
      SUPPORTED_LANGUAGES: typeof SUPPORTED_LANGUAGES;
    };
    // 直接暴露的便捷函数
    getDefaultEnglishSettings: typeof getDefaultEnglishSettings;
    getDefaultChineseSettings: typeof getDefaultChineseSettings;
    getDefaultConfig: typeof getDefaultConfig;
  }
}

if (typeof window !== 'undefined') {
  window.SubtitleConfig = {
    getDefaultEnglishSettings,
    getDefaultChineseSettings,
    getDefaultConfig,
    validateSettings,
    isEmptySettings,
    DEFAULT_API_CONFIG,
    SUPPORTED_MODELS,
    SUPPORTED_LANGUAGES,
  };

  // 直接暴露便捷函数（兼容旧代码）
  window.getDefaultEnglishSettings = getDefaultEnglishSettings;
  window.getDefaultChineseSettings = getDefaultChineseSettings;
  window.getDefaultConfig = getDefaultConfig;
}
