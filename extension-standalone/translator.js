/**
 * YouTube Subtitle Translator - 翻译服务模块
 * ============================================
 * 直接调用OpenAI兼容API进行字幕翻译
 * 实现完整流程：断句优化 → 内容总结 → 翻译
 */

// 语言代码映射
const LANGUAGE_MAPPING = {
  'zh': '简体中文',
  'zh-cn': '简体中文',
  'zh-tw': '繁体中文',
  'ja': '日文',
  'en': 'English',
  'ko': '韩文',
  'fr': '法文',
  'de': '德文',
  'es': '西班牙文',
  'pt': '葡萄牙文',
  'ru': '俄文'
};

// ========================================
// Prompt模板（和Python版一致）
// ========================================

// 断句优化Prompt
const SPLIT_SYSTEM_PROMPT = `You are an expert in subtitle editing and text formatting. Your role is to process subtitles for optimal readability and translation.

## Task
Process the provided subtitle text to make it suitable for translation and display. Each subtitle should be:
- Properly punctuated and complete sentences when possible
- Reasonably length (aim for 5-15 words per line)
- Free of filler words (um, uh, like, you know, etc.)
- Free of transcription artifacts ([music], [applause], etc.)

## Guidelines
1. Combine fragments into complete thoughts
2. Split overly long sentences at natural break points
3. Maintain the original meaning and tone
4. Preserve technical terms and proper nouns exactly
5. Remove non-speech sounds and annotations

## Output Format
Return a JSON object where each key is a number (starting from 1) mapping to the cleaned subtitle text:
{
  "1": "First cleaned subtitle text",
  "2": "Second cleaned subtitle text",
  ...
}

Return ONLY valid JSON, no other text or explanation.`;

// 内容总结Prompt
const SUMMARIZER_PROMPT = `You are an expert content analyst. Analyze the following subtitles and provide context information that will help with translation.

## Task
Extract key information about the content:
1. **Topic**: What is the main subject?
2. **Type**: Is this educational, entertainment, news, tutorial, interview, etc.?
3. **Formality**: Formal, casual, technical, conversational?
4. **Key Terms**: List important terms, names, or concepts that appear
5. **Context**: Any relevant background information

## Output Format
Return a JSON object:
{
  "topic": "Main topic description",
  "type": "Content type",
  "formality": "Formality level",
  "key_terms": ["term1", "term2", ...],
  "context": "Brief context description",
  "translation_notes": "Any special notes for translation"
}

Return ONLY valid JSON, no other text.`;

// 翻译Prompt
const TRANSLATE_PROMPT = `You are an expert specializing in subtitle proofreading and translation. Your role is to process subtitles generated through speech recognition and translate them into [TargetLanguage].

## Context Information
[ContextInfo]

## Processing Workflow

### 1. Subtitle Text Optimization
- Ensure subtitle numbering fully matches the input; do not combine, remove, or split subtitles.
- Correct spelling and grammar errors, ensure terminology is consistent.
- Eliminate filler words (e.g., "um," "uh," "like"), non-speech sound tags (e.g., [Music], [Applause]).
- If nothing remains after cleaning, set "optimized_subtitle" to an empty string.

### 2. Translation Procedures
- Using the cleaned and corrected original text, translate each subtitle into [TargetLanguage].
- Ensure contextual and technical accuracy in the translation.
- Preserve formatting, numbers, and symbols exactly.
- When translating technical terms, if a target language equivalent exists, translate it and keep the original term in parentheses. Example: "Generative AI" -> "生成式 AI (Generative AI)"
- Always translate each segment individually.

## Output Format
Return a valid JSON object where each key from the input maps to an object with the following structure:

\`\`\`json
{
  "subtitle_key": {
    "optimized_subtitle": "Cleaned and processed original text",
    "translation": "Translated text in [TargetLanguage]"
  }
}
\`\`\`

- Return strictly valid JSON with no extra fields, comments, or trailing commas.
- Output keys must exactly match the input keys.

## Standard Terminology (Do Not Change)
- AGI -> 通用人工智能
- LLM/Large Language Model -> 大语言模型
- Transformer -> Transformer
- Token -> Token
- Generative AI -> 生成式 AI
- AI Agent -> AI 智能体
- prompt -> 提示词
- fine-tuning -> 微调
- MCP (Model Context Protocol) -> MCP`;

/**
 * 翻译服务类 - 实现完整翻译流程
 */
class TranslatorService {
  constructor() {
    this.config = null;
    this.isTranslating = false;
    this.contextInfo = null; // 存储内容总结信息
  }

  /**
   * 加载API配置
   */
  async loadConfig() {
    return new Promise((resolve) => {
      chrome.storage.local.get(['apiConfig'], (result) => {
        this.config = result.apiConfig || {
          openaiBaseUrl: 'https://api.openai.com/v1',
          openaiApiKey: '',
          llmModel: 'gpt-4o-mini',
          targetLanguage: 'zh'
        };
        resolve(this.config);
      });
    });
  }

  /**
   * 获取目标语言名称
   */
  getTargetLanguageName(langCode) {
    const code = langCode.toLowerCase().trim();
    return LANGUAGE_MAPPING[code] || langCode;
  }

  /**
   * 调用OpenAI API
   * @param {string} systemPrompt - 系统提示
   * @param {string} userPrompt - 用户提示
   * @returns {Promise<string>} - API响应内容
   */
  async callOpenAI(systemPrompt, userPrompt) {
    await this.loadConfig();

    if (!this.config.openaiApiKey) {
      throw new Error('API密钥未配置');
    }

    const response = await fetch(`${this.config.openaiBaseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.config.openaiApiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: this.config.llmModel,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        temperature: 0.3
      })
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error?.message || `API请求失败: ${response.status}`);
    }

    const data = await response.json();
    return data.choices[0]?.message?.content || '';
  }

  /**
   * 解析JSON响应
   */
  parseJsonResponse(content) {
    try {
      // 尝试直接解析
      return JSON.parse(content);
    } catch (e) {
      // 尝试提取JSON块
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          return JSON.parse(jsonMatch[0]);
        } catch (e2) {
          console.error('JSON解析失败:', content);
          return null;
        }
      }
      return null;
    }
  }

  /**
   * 步骤1：断句优化
   * @param {Array<{start: number, end: number, text: string}>} subtitles - 原始字幕
   * @returns {Promise<Object>} - 优化后的字幕 {index: text}
   */
  async splitOptimize(subtitles) {
    console.log('📝 步骤1: 断句优化...');

    // 构建输入对象
    const inputObj = {};
    subtitles.forEach((sub, idx) => {
      inputObj[String(idx + 1)] = sub.text;
    });

    const response = await this.callOpenAI(
      SPLIT_SYSTEM_PROMPT,
      `Please process these subtitles:\n\n${JSON.stringify(inputObj, null, 2)}`
    );

    const result = this.parseJsonResponse(response);
    if (!result) {
      console.warn('断句优化解析失败，使用原始字幕');
      return inputObj;
    }

    console.log('✅ 断句优化完成');
    return result;
  }

  /**
   * 步骤2：内容总结
   * @param {Object} optimizedSubtitles - 优化后的字幕 {index: text}
   * @returns {Promise<Object>} - 内容上下文信息
   */
  async summarizeContent(optimizedSubtitles) {
    console.log('📊 步骤2: 内容总结...');

    // 提取前20条字幕用于总结（避免超长输入）
    const keys = Object.keys(optimizedSubtitles).slice(0, 20);
    const sampleText = keys.map(k => optimizedSubtitles[k]).join('\n');

    const response = await this.callOpenAI(
      SUMMARIZER_PROMPT,
      `Analyze these subtitles and provide context:\n\n${sampleText}`
    );

    const result = this.parseJsonResponse(response);
    if (!result) {
      console.warn('内容总结解析失败，使用默认上下文');
      return {
        topic: 'Unknown',
        type: 'Video content',
        formality: 'Neutral',
        key_terms: [],
        context: 'No context available',
        translation_notes: ''
      };
    }

    this.contextInfo = result;
    console.log('✅ 内容总结完成:', result.topic);
    return result;
  }

  /**
   * 步骤3：翻译
   * @param {Object} optimizedSubtitles - 优化后的字幕 {index: text}
   * @param {Object} contextInfo - 内容上下文信息
   * @param {string} targetLang - 目标语言代码
   * @param {Function} onProgress - 进度回调
   * @returns {Promise<Object>} - 翻译结果 {index: {optimized_subtitle, translation}}
   */
  async translate(optimizedSubtitles, contextInfo, targetLang, onProgress) {
    console.log('🌐 步骤3: 翻译字幕...');

    const targetLanguageName = this.getTargetLanguageName(targetLang);

    // 构建上下文信息字符串
    const contextStr = `
Topic: ${contextInfo.topic || 'Unknown'}
Type: ${contextInfo.type || 'Video'}
Formality: ${contextInfo.formality || 'Neutral'}
Key Terms: ${(contextInfo.key_terms || []).join(', ')}
Context: ${contextInfo.context || ''}
Notes: ${contextInfo.translation_notes || ''}`.trim();

    // 构建翻译Prompt
    const translationPrompt = TRANSLATE_PROMPT
      .replace(/\[TargetLanguage\]/g, targetLanguageName)
      .replace('[ContextInfo]', contextStr);

    const keys = Object.keys(optimizedSubtitles);
    const batchSize = 15;
    const results = {};

    for (let i = 0; i < keys.length; i += batchSize) {
      const batchKeys = keys.slice(i, i + batchSize);
      const batchObj = {};
      batchKeys.forEach(k => {
        batchObj[k] = optimizedSubtitles[k];
      });

      const response = await this.callOpenAI(
        translationPrompt,
        `Translate these subtitles:\n\n${JSON.stringify(batchObj, null, 2)}`
      );

      const batchResult = this.parseJsonResponse(response);
      if (batchResult) {
        Object.assign(results, batchResult);
      } else {
        // 回退：对未解析的项使用原文
        batchKeys.forEach(k => {
          results[k] = {
            optimized_subtitle: optimizedSubtitles[k],
            translation: optimizedSubtitles[k]
          };
        });
      }

      // 进度回调
      if (onProgress) {
        onProgress(Math.min(i + batchSize, keys.length), keys.length);
      }

      // 延迟避免限流
      if (i + batchSize < keys.length) {
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }

    console.log('✅ 翻译完成');
    return results;
  }

  /**
   * 执行完整翻译流程
   * @param {Array<{start: number, end: number, text: string}>} subtitles - 原始字幕数组
   * @param {string} targetLang - 目标语言代码
   * @param {Function} onProgress - 进度回调 (step, current, total)
   * @returns {Promise<{english: Array, chinese: Array}>} - 翻译后的双语字幕
   */
  async translateFull(subtitles, targetLang = 'zh', onProgress = null) {
    if (this.isTranslating) {
      throw new Error('翻译正在进行中');
    }

    this.isTranslating = true;

    // 保存翻译状态到 storage
    const saveProgress = async (step, current, total) => {
      await chrome.storage.local.set({
        translationProgress: {
          isTranslating: true,
          step,
          current,
          total,
          timestamp: Date.now()
        }
      });
      if (onProgress) onProgress(step, current, total);
    };

    try {
      await this.loadConfig();

      // 步骤1：断句优化
      await saveProgress('split', 0, 3);
      const optimizedSubtitles = await this.splitOptimize(subtitles);
      await saveProgress('split', 1, 3);

      // 步骤2：内容总结
      await saveProgress('summary', 1, 3);
      const contextInfo = await this.summarizeContent(optimizedSubtitles);
      await saveProgress('summary', 2, 3);

      // 步骤3：翻译
      await saveProgress('translate', 2, 3);
      const translations = await this.translate(
        optimizedSubtitles,
        contextInfo,
        targetLang,
        async (current, total) => {
          const progress = 2 + (current / total);
          await saveProgress('translate', progress, 3);
        }
      );
      await saveProgress('complete', 3, 3);

      // 构建结果数组
      const englishSubtitles = [];
      const chineseSubtitles = [];

      subtitles.forEach((sub, idx) => {
        const key = String(idx + 1);
        const result = translations[key] || {};

        englishSubtitles.push({
          startTime: sub.startTime,
          endTime: sub.endTime,
          text: result.optimized_subtitle || sub.text
        });

        chineseSubtitles.push({
          startTime: sub.startTime,
          endTime: sub.endTime,
          text: result.translation || ''
        });
      });

      return { english: englishSubtitles, chinese: chineseSubtitles };

    } finally {
      this.isTranslating = false;
      // 清除翻译进度状态
      chrome.storage.local.remove('translationProgress');
    }
  }

  /**
   * 取消翻译
   */
  cancelTranslation() {
    this.isTranslating = false;
  }
}

// 创建全局实例
const translatorService = new TranslatorService();

// 导出（兼容不同环境）
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { TranslatorService, translatorService };
} else if (typeof window !== 'undefined') {
  window.TranslatorService = TranslatorService;
  window.translatorService = translatorService;
}
