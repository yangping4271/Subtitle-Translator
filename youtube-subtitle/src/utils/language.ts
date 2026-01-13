/**
 * 语言代码映射 - 与 Python 版本 config.py 一致
 */

export const LANGUAGE_MAPPING: Record<string, string> = {
  'zh': '简体中文',
  'zh-cn': '简体中文',
  'zh-tw': '繁体中文',
  'ja': '日文',
  'japanese': '日文',
  'en': 'English',
  'english': 'English',
  'ko': '韩文',
  'korean': '韩文',
  'fr': '法文',
  'french': '法文',
  'de': '德文',
  'german': '德文',
  'es': '西班牙文',
  'spanish': '西班牙文',
  'pt': '葡萄牙文',
  'portuguese': '葡萄牙文',
  'it': '意大利文',
  'italian': '意大利文',
  'ru': '俄文',
  'russian': '俄文',
  'ar': '阿拉伯文',
  'arabic': '阿拉伯文',
  'th': '泰文',
  'thai': '泰文',
  'vi': '越南文',
  'vietnamese': '越南文',
};

/**
 * 获取语言名称
 */
export function getLanguageName(langCode: string): string {
  const code = langCode.toLowerCase().trim();
  return LANGUAGE_MAPPING[code] || langCode;
}

/**
 * 支持的目标语言列表
 */
export const SUPPORTED_LANGUAGES = Object.keys(LANGUAGE_MAPPING);
