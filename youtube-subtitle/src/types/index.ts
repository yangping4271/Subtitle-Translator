/**
 * 字幕相关类型定义
 *
 * 时间单位标准：所有时间戳使用毫秒（整数）
 * - 与 Python 版本保持一致
 * - SRT 解析后直接存储为毫秒
 * - 内部计算和比较均使用毫秒
 */

/** 原始字幕条目 */
export interface SubtitleEntry {
  index: number;
  startTime: number;  // 毫秒（整数）
  endTime: number;    // 毫秒（整数）
  text: string;
}

/** 翻译后的字幕条目 */
export interface TranslatedEntry {
  index: number;
  startTime: number;  // 毫秒（整数）
  endTime: number;    // 毫秒（整数）
  original: string;
  optimized: string;
  translation: string;
}

/** 双语字幕结果 */
export interface BilingualSubtitles {
  english: SubtitleEntry[];
  chinese: SubtitleEntry[];
}

/** 断句结果 */
export interface SplitResult {
  segments: string[];
  stats: SplitStats;
}

/** 断句统计 */
export interface SplitStats {
  normal: number;      // ≤ target
  tolerated: number;   // target < x ≤ tolerance
  optimized: number;   // tolerance < x ≤ warning
  forced: number;      // warning < x ≤ max
  rejected: number;    // > max
}

/** 总结结果 */
export interface SummaryResult {
  context: {
    type: string;
    topic: string;
    formality: string;
  };
  corrections: Record<string, string>;
  canonical_terms: string[];
  do_not_translate: string[];
  style_guide: {
    audience: string;
    technical_level: string;
    tone: string;
  };
}

/** API 配置 */
export interface ApiConfig {
  openaiBaseUrl: string;
  openaiApiKey: string;
  llmModel: string;
  targetLanguage: string;
}

/** 翻译器配置 */
export interface TranslatorConfig {
  openaiBaseUrl: string;
  openaiApiKey: string;
  splitModel: string;
  summaryModel: string;
  translationModel: string;
  targetLanguage: string;
  maxWordCountEnglish: number;
  threadNum: number;
  batchSize: number;
  // 阈值倍数
  toleranceMultiplier: number;
  warningMultiplier: number;
  maxMultiplier: number;
}

/** 翻译进度回调 */
export type ProgressCallback = (
  step: 'split' | 'summary' | 'translate' | 'complete',
  current: number,
  total: number
) => void;

/** 翻译选项 */
export interface TranslateOptions {
  inputFile?: string;
  videoTitle?: string;
  debug?: boolean;
  onProgress?: ProgressCallback;
}

/** LLM API 响应 */
export interface LLMResponse {
  choices: Array<{
    message: {
      content: string;
    };
  }>;
}

/** 批次翻译结果 */
export interface BatchTranslateResult {
  optimized_subtitles: Record<string, string>;
  translated_subtitles: Record<string, string | TranslationDetail>;
}

/** 翻译详情（包含修订） */
export interface TranslationDetail {
  translation: string;
  revised_translation?: string;
  revise_suggestions?: string;
}

// ========================================
// Chrome Extension 相关类型
// ========================================

/** 字幕样式设置 */
export interface SubtitleStyleSettings {
  fontSize: number;
  fontColor: string;
  fontFamily: string;
  fontWeight: string;
  textStroke: string;
  textShadow: string;
  lineHeight: number;
}

/** DPR 补偿配置 */
export interface DPRConfig {
  enabled: boolean;
  compensationFactor: number;
}

/** UI 配置 */
export interface UIConfig {
  fontSizeMin: number;
  fontSizeMax: number;
}

/** 完整字幕配置 */
export interface SubtitleConfig {
  english: SubtitleStyleSettings;
  chinese: SubtitleStyleSettings;
  dpr: DPRConfig;
  ui: UIConfig;
}

/** 支持的模型选项 */
export interface ModelOption {
  value: string;
  text: string;
}

/** 支持的语言选项 */
export interface LanguageOption {
  value: string;
  text: string;
}

/** 简单字幕条目（不含 index） */
export interface SimpleSubtitleEntry {
  startTime: number;
  endTime: number;
  text: string;
}

/** ASS 解析结果 */
export interface ASSParseResult {
  english: SimpleSubtitleEntry[];
  chinese: SimpleSubtitleEntry[];
}

/** 视频信息 */
export interface VideoInfo {
  ytTitle: string;
  channelName: string;
  uploadDate: string;
  videoURL: string;
  videoId: string;
}

/** 视频字幕缓存数据 */
export interface VideoSubtitleData {
  videoId: string;
  timestamp: string;
  englishSubtitles?: SimpleSubtitleEntry[];
  chineseSubtitles?: SimpleSubtitleEntry[];
  subtitleData?: SimpleSubtitleEntry[];
  englishFileName?: string;
  chineseFileName?: string;
  fileName?: string;
}

/** 翻译进度状态 */
export interface TranslationProgress {
  isTranslating: boolean;
  step?: string;
  current?: number;
  total?: number;
  error?: string;
  timestamp?: number;
}

/** Chrome 消息类型 */
export type ChromeMessageAction =
  | 'getSubtitleData'
  | 'getBilingualSubtitleData'
  | 'saveSubtitleData'
  | 'saveBilingualSubtitles'
  | 'saveVideoSubtitles'
  | 'toggleSubtitle'
  | 'updateSettings'
  | 'clearSubtitleData'
  | 'forceReset'
  | 'setSubtitleEnabled'
  | 'startTranslation'
  | 'cancelTranslation'
  | 'getTranslationStatus'
  | 'loadSubtitle'
  | 'loadBilingualSubtitles'
  | 'clearData'
  | 'getVideoInfo'
  | 'getSubtitleStatus'
  | 'getYouTubeSubtitles'
  | 'printEnglishSubtitles'
  | 'toggleAutoLoad'
  | 'autoLoadSuccess'
  | 'autoLoadError';
