/**
 * YouTube Subtitle Translator (Standalone) - 统一配置中心
 * ============================================
 * 所有默认值的单一数据源 (Single Source of Truth)
 *
 * 修改任何默认配置只需要在这个文件中修改一次即可，
 * 所有其他模块会自动使用更新后的值。
 */

/**
 * 默认API配置
 * @type {Object}
 */
const DEFAULT_API_CONFIG = {
  openaiBaseUrl: 'https://api.openai.com/v1',
  openaiApiKey: '',
  llmModel: 'gpt-4o-mini',
  targetLanguage: 'zh'
};

/**
 * 支持的LLM模型列表
 */
const SUPPORTED_MODELS = [
  { value: 'gpt-4o-mini', text: 'GPT-4o Mini (推荐)' },
  { value: 'gpt-4o', text: 'GPT-4o' },
  { value: 'gpt-4-turbo', text: 'GPT-4 Turbo' },
  { value: 'gpt-3.5-turbo', text: 'GPT-3.5 Turbo' },
  { value: 'claude-3-haiku-20240307', text: 'Claude 3 Haiku' },
  { value: 'claude-3-sonnet-20240229', text: 'Claude 3 Sonnet' },
  { value: 'google/gemini-3-flash-preview', text: 'Gemini 3 Flash (Preview)' },
  { value: 'google/gemini-flash-1.5', text: 'Gemini 1.5 Flash' },
  { value: 'google/gemini-pro-1.5', text: 'Gemini 1.5 Pro' },
  { value: 'custom', text: '自定义模型...' }
];

/**
 * 支持的目标语言列表
 */
const SUPPORTED_LANGUAGES = [
  { value: 'zh', text: '简体中文' },
  { value: 'zh-tw', text: '繁体中文' },
  { value: 'ja', text: '日文' },
  { value: 'ko', text: '韩文' },
  { value: 'en', text: 'English' },
  { value: 'fr', text: '法文' },
  { value: 'de', text: '德文' },
  { value: 'es', text: '西班牙文' }
];

/**
 * 获取默认API配置的深拷贝
 */
function getDefaultApiConfig() {
  return JSON.parse(JSON.stringify(DEFAULT_API_CONFIG));
}

/**
 * 默认字幕配置
 * @type {Object}
 */
const DEFAULT_SUBTITLE_CONFIG = {
  /**
   * 英文字幕默认配置
   * 使用 Noto Serif 衬线字体，适合长时间阅读
   */
  english: {
    fontSize: 30,                                  // 字体大小(px)，会根据 DPR 自动补偿
    fontColor: '#FFFF00',                          // 字体颜色（黄色）
    fontFamily: '"Noto Serif", Georgia, serif',   // 字体族（优先 Noto Serif）
    fontWeight: '700',                             // 字体粗细（粗体）
    textStroke: '2px #000000',                     // 文字描边（2px 黑色）
    textShadow: 'none',                            // 文字阴影（已禁用）
    lineHeight: 1.3                                // 行高倍数
  },

  /**
   * 中文字幕默认配置
   * 使用宋体，更适合中文显示
   */
  chinese: {
    fontSize: 28,                                  // 字体大小(px)，会根据 DPR 自动补偿
    fontColor: '#00FF00',                          // 字体颜色（绿色）
    fontFamily: '"Songti SC", serif',             // 字体族（宋体）
    fontWeight: '900',                             // 字体粗细（更粗）
    textStroke: '2px #000000',                     // 文字描边（2px 黑色）
    textShadow: 'none',                            // 文字阴影（已禁用）
    lineHeight: 1.4                                // 行高倍数（中文略高）
  },

  /**
   * DPR (设备像素比) 自动补偿配置
   * 在高分辨率屏幕（Retina、4K）上自动放大字体以匹配视觉大小
   */
  dpr: {
    enabled: true,          // 是否启用 DPR 补偿
    compensationFactor: 0.4 // 补偿系数，公式：1 + (DPR - 1) × factor
    // DPR=1: 1.0x (不补偿)
    // DPR=2: 1.4x (放大 40%)
    // DPR=3: 1.8x (放大 80%)
  },

  /**
   * UI 配置
   */
  ui: {
    fontSizeMin: 16,  // 字体大小滑块最小值
    fontSizeMax: 48   // 字体大小滑块最大值
  }
};

/**
 * 获取英文字幕默认配置的深拷贝
 *
 * @returns {Object} 英文字幕配置对象
 *
 * @example
 * const settings = getDefaultEnglishSettings();
 * settings.fontSize = 40; // 修改不会影响原始配置
 */
function getDefaultEnglishSettings() {
  return JSON.parse(JSON.stringify(DEFAULT_SUBTITLE_CONFIG.english));
}

/**
 * 获取中文字幕默认配置的深拷贝
 *
 * @returns {Object} 中文字幕配置对象
 *
 * @example
 * const settings = getDefaultChineseSettings();
 * settings.fontSize = 35; // 修改不会影响原始配置
 */
function getDefaultChineseSettings() {
  return JSON.parse(JSON.stringify(DEFAULT_SUBTITLE_CONFIG.chinese));
}

/**
 * 获取完整的默认配置（只读）
 *
 * @returns {Object} 完整配置对象（包含 english、chinese、dpr、server、ui）
 *
 * @example
 * const config = getDefaultConfig();
 * console.log(config.dpr.enabled); // true
 * console.log(config.server.defaultUrl); // 'http://127.0.0.1:8888'
 */
function getDefaultConfig() {
  return DEFAULT_SUBTITLE_CONFIG;
}

/**
 * 验证和补全设置对象
 * 确保所有必需的属性都存在，缺失的使用默认值
 *
 * @param {Object} settings - 要验证的设置对象
 * @param {string} type - 设置类型 ('english' 或 'chinese')
 * @returns {Object} 验证后的完整设置对象
 *
 * @example
 * const userSettings = { fontSize: 40 };
 * const validated = validateSettings(userSettings, 'english');
 * // validated 包含所有属性，缺失的使用默认值
 */
function validateSettings(settings, type) {
  const defaults = type === 'english'
    ? getDefaultEnglishSettings()
    : getDefaultChineseSettings();

  const validated = {};
  for (const key in defaults) {
    validated[key] = settings && settings[key] !== undefined
      ? settings[key]
      : defaults[key];
  }

  return validated;
}

/**
 * 检查设置对象是否为空
 *
 * @param {Object} obj - 要检查的对象
 * @returns {boolean} 如果对象为空或不存在返回 true
 */
function isEmptySettings(obj) {
  return !obj || Object.keys(obj).length === 0;
}

// ===========================================
// 导出配置和工具函数
// ===========================================

// 兼容不同的模块系统
if (typeof module !== 'undefined' && module.exports) {
  // Node.js / CommonJS
  module.exports = {
    DEFAULT_SUBTITLE_CONFIG,
    getDefaultEnglishSettings,
    getDefaultChineseSettings,
    getDefaultConfig,
    validateSettings,
    isEmptySettings
  };
} else if (typeof window !== 'undefined') {
  // 浏览器环境 - 挂载到全局
  window.SubtitleConfig = {
    getDefaultEnglishSettings,
    getDefaultChineseSettings,
    getDefaultConfig,
    validateSettings,
    isEmptySettings
  };
}
