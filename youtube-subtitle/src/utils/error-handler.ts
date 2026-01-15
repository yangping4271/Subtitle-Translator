/**
 * 错误处理工具
 */

/**
 * 错误类型枚举
 */
export enum ErrorCategory {
  MODEL_NOT_FOUND = 'model_not_found',
  INVALID_API_KEY = 'invalid_api_key',
  RATE_LIMIT = 'rate_limit',
  TIMEOUT = 'timeout',
  NETWORK = 'network',
  SERVER_ERROR = 'server_error',
  UNKNOWN = 'unknown',
}

/**
 * 错误信息接口
 */
export interface ErrorInfo {
  category: ErrorCategory;
  message: string;
  suggestion: string;
  isRetryable: boolean;
}

/**
 * 错误分类规则
 */
const ERROR_RULES: Array<{
  category: ErrorCategory;
  patterns: string[];
  suggestion: string;
  isRetryable: boolean;
}> = [
  {
    category: ErrorCategory.MODEL_NOT_FOUND,
    patterns: ['model_not_found', 'model does not exist'],
    suggestion: '请检查模型名称是否正确，或更换其他可用模型（如 gpt-4o-mini, gpt-4o）',
    isRetryable: false,
  },
  {
    category: ErrorCategory.INVALID_API_KEY,
    patterns: ['invalid api key', 'incorrect api key', 'api key not found', 'authentication failed', 'invalid_api_key'],
    suggestion: '请检查 API 密钥是否正确配置，确认密钥有效且未过期',
    isRetryable: false,
  },
  {
    category: ErrorCategory.RATE_LIMIT,
    patterns: ['rate_limit_exceeded', 'rate limit', 'too many requests', '429'],
    suggestion: '请稍后重试，或考虑升级 API 套餐以获得更高的速率限制',
    isRetryable: true,
  },
  {
    category: ErrorCategory.TIMEOUT,
    patterns: ['timeout', 'timed out', 'aborterror'],
    suggestion: '请检查网络连接，或稍后重试。如果问题持续，请考虑减小批次大小',
    isRetryable: true,
  },
  {
    category: ErrorCategory.NETWORK,
    patterns: ['network', 'fetch failed', 'econnreset', 'econnrefused'],
    suggestion: '请检查网络连接，确认能够访问 OpenAI API 服务器',
    isRetryable: true,
  },
  {
    category: ErrorCategory.SERVER_ERROR,
    patterns: ['500', '502', '503', 'internal server error', 'bad gateway', 'service unavailable'],
    suggestion: '服务器暂时不可用，请稍后重试',
    isRetryable: true,
  },
];

/**
 * 提取错误消息
 */
export function extractErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;

  if (typeof error === 'object' && error !== null) {
    const apiError = error as { error?: { message?: string } };
    if (apiError.error?.message) return apiError.error.message;

    try {
      return JSON.stringify(error);
    } catch {
      return String(error);
    }
  }

  return String(error);
}

/**
 * 分类错误并提供建议
 */
export function classifyErrorWithSuggestion(error: unknown): ErrorInfo {
  const message = extractErrorMessage(error).toLowerCase();

  for (const rule of ERROR_RULES) {
    if (rule.patterns.some(pattern => message.includes(pattern))) {
      return {
        category: rule.category,
        message: extractErrorMessage(error),
        suggestion: `💡 建议：${rule.suggestion}`,
        isRetryable: rule.isRetryable,
      };
    }
  }

  return {
    category: ErrorCategory.UNKNOWN,
    message: extractErrorMessage(error),
    suggestion: '💡 建议：请查看错误详情，或联系技术支持',
    isRetryable: true,
  };
}

/**
 * 格式化错误信息，包含建议
 */
export function formatErrorWithSuggestion(error: unknown): string {
  const errorInfo = classifyErrorWithSuggestion(error);
  return `${errorInfo.message}\n${errorInfo.suggestion}`;
}

/**
 * 自定义翻译错误类
 */
export class TranslationError extends Error {
  public readonly suggestion: string;
  public readonly category: ErrorCategory;
  public readonly isRetryable: boolean;

  constructor(message: string, errorInfo?: Partial<ErrorInfo>) {
    super(message);
    this.name = 'TranslationError';
    this.suggestion = errorInfo?.suggestion || '请检查配置或稍后重试';
    this.category = errorInfo?.category || ErrorCategory.UNKNOWN;
    this.isRetryable = errorInfo?.isRetryable ?? true;
  }

  static fromError(error: unknown, context?: string): TranslationError {
    const errorInfo = classifyErrorWithSuggestion(error);
    const message = context ? `${context}: ${errorInfo.message}` : errorInfo.message;
    return new TranslationError(message, errorInfo);
  }

  getFullMessage(): string {
    return `${this.message}\n${this.suggestion}`;
  }
}
