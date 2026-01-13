/**
 * JSON 修复模块 - 移植自 Python json_repair.py
 * 用于修复 LLM 返回的不完整或格式错误的 JSON
 */

import { setupLogger } from './logger.js';

const logger = setupLogger('json_repair');

type JSONValue = string | number | boolean | null | JSONValue[] | { [key: string]: JSONValue };

/**
 * JSON 解析器类 - 实现宽松的 JSON 解析
 */
class JSONParser {
  private json_str: string;
  private index: number;
  private context: string[];

  constructor(json_str: string) {
    this.json_str = json_str;
    this.index = 0;
    this.context = [];
  }

  parse(): JSONValue {
    const result = this.parseJson();
    return result;
  }

  private parseJson(): JSONValue {
    while (true) {
      const char = this.getCharAt();

      if (char === false) {
        return '';
      }

      const isInContext = this.context.length > 0;

      if (char === '{') {
        this.index++;
        return this.parseObject();
      } else if (char === '[') {
        this.index++;
        return this.parseArray();
      } else if (char === '}') {
        return '';
      } else if (isInContext && (char === '"' || char === "'" || /[a-zA-Z]/.test(char))) {
        return this.parseString();
      } else if (isInContext && (/[0-9]/.test(char) || char === '-' || char === '.')) {
        return this.parseNumber();
      } else {
        this.index++;
      }
    }
  }

  private parseObject(): { [key: string]: JSONValue } {
    const obj: { [key: string]: JSONValue } = {};

    while ((this.getCharAt() || '}') !== '}') {
      this.skipWhitespaces();

      if ((this.getCharAt() || '') === ':') {
        this.index++;
      }

      this.setContext('object_key');
      this.skipWhitespaces();

      let key = '';
      while (key === '' && this.getCharAt()) {
        const currentIndex = this.index;
        key = this.parseString() as string;

        if (key === '' && this.getCharAt() === ':') {
          key = 'empty_placeholder';
          break;
        } else if (key === '' && this.index === currentIndex) {
          this.index++;
        }
      }

      this.skipWhitespaces();

      if ((this.getCharAt() || '}') === '}') {
        continue;
      }

      this.skipWhitespaces();

      if ((this.getCharAt() || '') !== ':') {
        // Missing colon after key
      }

      this.index++;
      this.resetContext();
      this.setContext('object_value');

      const value = this.parseJson();
      this.resetContext();
      obj[key] = value;

      const nextChar = this.getCharAt() || '';
      if (nextChar === ',' || nextChar === "'" || nextChar === '"') {
        this.index++;
      }

      this.skipWhitespaces();
    }

    this.index++;
    return obj;
  }

  private parseArray(): JSONValue[] {
    const arr: JSONValue[] = [];
    this.setContext('array');

    while ((this.getCharAt() || ']') !== ']') {
      this.skipWhitespaces();
      const value = this.parseJson();

      if (value === '') {
        break;
      }

      if (value !== '...') {
        arr.push(value);
      }

      let char = this.getCharAt();
      while (char && (/\s/.test(char) || char === ',')) {
        this.index++;
        char = this.getCharAt();
      }

      if (this.getContext() === 'object_value' && char === '}') {
        break;
      }
    }

    this.index++;
    this.resetContext();
    return arr;
  }

  private parseString(): string | JSONValue {
    let missingQuotes = false;
    let stringDelimiter = '"';

    let char = this.getCharAt();

    while (char && char !== '"' && char !== "'" && !/[a-zA-Z0-9]/.test(char)) {
      this.index++;
      char = this.getCharAt();
    }

    if (!char) {
      return '';
    }

    if (char === "'") {
      stringDelimiter = "'";
    } else if (/[a-zA-Z]/.test(char)) {
      // Check for boolean/null
      if (['t', 'f', 'n'].includes(char.toLowerCase()) && this.getContext() !== 'object_key') {
        const value = this.parseBooleanOrNull();
        if (value !== '') {
          return value;
        }
      }
      missingQuotes = true;
    }

    if (!missingQuotes) {
      this.index++;
    }

    let stringAcc = '';

    char = this.getCharAt();
    while (char && char !== stringDelimiter) {
      if (missingQuotes) {
        if (this.getContext() === 'object_key' && (char === ':' || /\s/.test(char))) {
          break;
        } else if (this.getContext() === 'object_value' && (char === ',' || char === '}')) {
          break;
        }
      }

      stringAcc += char;
      this.index++;
      char = this.getCharAt();

      // Handle escape sequences
      if (stringAcc.length > 1 && stringAcc[stringAcc.length - 1] === '\\') {
        stringAcc = stringAcc.slice(0, -1);
        if (char && [stringDelimiter, 't', 'n', 'r', 'b', '\\'].includes(char)) {
          const escapeSeqs: Record<string, string> = { t: '\t', n: '\n', r: '\r', b: '\b' };
          stringAcc += escapeSeqs[char] || char;
          this.index++;
          char = this.getCharAt();
        }
      }
    }

    if (char !== stringDelimiter) {
      // Missing closing quote
    } else {
      this.index++;
    }

    return stringAcc.trimEnd();
  }

  private parseNumber(): number | string {
    let numberStr = '';
    const numberChars = new Set('0123456789-.eE/,');
    let char = this.getCharAt();
    const isArray = this.getContext() === 'array';

    while (char && numberChars.has(char) && (char !== ',' || !isArray)) {
      numberStr += char;
      this.index++;
      char = this.getCharAt();
    }

    if (numberStr.length > 1 && ['-', 'e', 'E', '/', ','].includes(numberStr[numberStr.length - 1])) {
      numberStr = numberStr.slice(0, -1);
      this.index--;
    }

    if (numberStr) {
      if (numberStr.includes(',')) {
        return numberStr;
      }
      if (numberStr.includes('.') || numberStr.includes('e') || numberStr.includes('E')) {
        const parsed = parseFloat(numberStr);
        return isNaN(parsed) ? numberStr : parsed;
      } else if (numberStr === '-') {
        return this.parseJson() as number;
      } else {
        const parsed = parseInt(numberStr, 10);
        return isNaN(parsed) ? numberStr : parsed;
      }
    }

    return this.parseJson() as number;
  }

  private parseBooleanOrNull(): boolean | null | string {
    const startingIndex = this.index;
    const char = (this.getCharAt() || '').toLowerCase();

    let value: [string, boolean | null] | undefined;
    if (char === 't') {
      value = ['true', true];
    } else if (char === 'f') {
      value = ['false', false];
    } else if (char === 'n') {
      value = ['null', null];
    }

    if (value) {
      let i = 0;
      let c = (this.getCharAt() || '').toLowerCase();
      while (c && i < value[0].length && c === value[0][i]) {
        i++;
        this.index++;
        c = (this.getCharAt() || '').toLowerCase();
      }
      if (i === value[0].length) {
        return value[1];
      }
    }

    this.index = startingIndex;
    return '';
  }

  private getCharAt(offset = 0): string | false {
    const idx = this.index + offset;
    if (idx >= 0 && idx < this.json_str.length) {
      return this.json_str[idx];
    }
    return false;
  }

  private skipWhitespaces(): void {
    while (this.index < this.json_str.length && /\s/.test(this.json_str[this.index])) {
      this.index++;
    }
  }

  private setContext(value: string): void {
    if (value) {
      this.context.push(value);
    }
  }

  private resetContext(): void {
    this.context.pop();
  }

  private getContext(): string {
    return this.context[this.context.length - 1] || '';
  }
}

/**
 * 修复 JSON 字符串
 */
export function repairJson(jsonStr: string): JSONValue {
  const parser = new JSONParser(jsonStr);
  try {
    return JSON.parse(jsonStr);
  } catch {
    return parser.parse();
  }
}

/**
 * 类似 JSON.parse 的函数，但会尝试修复无效 JSON
 */
export function loads(jsonStr: string): JSONValue {
  return repairJson(jsonStr);
}

/**
 * 清理 LLM 返回的 JSON 字符串
 */
export function cleanLlmResponse(response: string): string {
  // 移除开头和结尾的三引号和换行符
  let cleaned = response.trim().replace(/^["'\n\s]+|["'\n\s]+$/g, '');

  if (cleaned.startsWith('"""') && cleaned.endsWith('"""')) {
    cleaned = cleaned.slice(3, -3);
  }

  // 移除可能的转义字符
  cleaned = cleaned.replace(/\\"/g, '"');

  return cleaned.trim();
}

/**
 * 解析 LLM 返回的 JSON 响应
 * @param response LLM 返回的原始响应字符串
 * @returns 解析后的 JSON 对象，如果解析失败则返回空对象
 */
export function parseLlmResponse(response: string): Record<string, unknown> {
  // 0. 首先移除 <think></think> 标签
  let cleaned = response;
  cleaned = cleaned.replace(/<think>[\s\S]*?<\/think>/g, '');
  // 清理可能的空行
  cleaned = cleaned.replace(/\n+/g, '\n');

  // 1. 尝试清理三引号并直接解析
  cleaned = cleanLlmResponse(cleaned);

  // 2. 尝试提取 JSON 块（处理 markdown 代码块）
  const jsonBlockMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (jsonBlockMatch) {
    cleaned = jsonBlockMatch[1].trim();
  }

  // 3. 尝试直接解析
  try {
    const result = JSON.parse(cleaned);
    if (typeof result === 'object' && result !== null) {
      return result as Record<string, unknown>;
    }
    return {};
  } catch {
    // 继续尝试修复
  }

  // 4. 使用 JSON 修复器
  try {
    const result = loads(cleaned);
    if (typeof result === 'object' && result !== null && !Array.isArray(result)) {
      return result as Record<string, unknown>;
    }
    if (Array.isArray(result)) {
      return result as unknown as Record<string, unknown>;
    }
    return {};
  } catch {
    // 继续尝试修复
  }

  // 5. 尝试修复常见的 JSON 格式错误
  try {
    let fixedJson = cleaned;
    // 修复错误的逗号
    fixedJson = fixedJson.replace(/,\s*}/g, '}');
    fixedJson = fixedJson.replace(/,\s*]/g, ']');
    // 修复缺少引号的键
    fixedJson = fixedJson.replace(/(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:/g, '$1"$2":');

    const result = JSON.parse(fixedJson);
    if (typeof result === 'object' && result !== null) {
      return result as Record<string, unknown>;
    }
    return {};
  } catch (e) {
    logger.error(`JSON 解析失败: ${e}`);
    logger.warn(`原始响应片段: ${response.slice(0, 100)}...`);
    return {};
  }
}
