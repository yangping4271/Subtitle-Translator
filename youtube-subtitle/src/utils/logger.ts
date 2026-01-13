/**
 * 日志系统 - 环境感知，与 Python 版本格式一致
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  level: LogLevel;
  module: string;
  message: string;
  timestamp: string;
  data?: unknown;
}

// 检测运行环境
const isNode = typeof process !== 'undefined' && process.versions?.node;
const isBrowser = typeof window !== 'undefined';

// 日志文件支持（仅 CLI 模式）
let fileLoggingEnabled = false;
let logFilePath: string | null = null;

/**
 * 初始化文件日志（仅 CLI 调用）
 */
export async function initFileLogging(logDir: string, filename = 'cli.log'): Promise<void> {
  if (!isNode) return;

  try {
    const fs = await import('fs');
    const path = await import('path');

    // 确保 log 目录存在
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }

    logFilePath = path.join(logDir, filename);

    // 清空旧日志文件（每次运行覆盖）
    fs.writeFileSync(logFilePath, '', 'utf-8');

    fileLoggingEnabled = true;
    console.log(`📝 日志文件: ${logFilePath}`);
  } catch (error) {
    console.error('❌ 无法创建日志文件:', error);
  }
}

// 日志格式化（与 Python 版本一致）
function formatLog(entry: LogEntry): string {
  const time = entry.timestamp.split('T')[1].split('.')[0]; // HH:MM:SS
  return `${time} [${entry.module}] ${entry.message}`;
}

// Node.js 终端颜色
const colors = {
  debug: '\x1b[36m',  // cyan
  info: '\x1b[32m',   // green
  warn: '\x1b[33m',   // yellow
  error: '\x1b[31m',  // red
  reset: '\x1b[0m',
};

export class Logger {
  private module: string;
  private debugEnabled: boolean;

  constructor(module: string, debugEnabled = false) {
    this.module = module;
    this.debugEnabled = debugEnabled;
  }

  setDebug(enabled: boolean): void {
    this.debugEnabled = enabled;
  }

  private log(level: LogLevel, message: string, data?: unknown): void {
    if (level === 'debug' && !this.debugEnabled) {
      return;
    }

    const entry: LogEntry = {
      level,
      module: this.module,
      message,
      timestamp: new Date().toISOString(),
      data,
    };

    const formatted = formatLog(entry);

    if (isNode) {
      // Node.js 环境：彩色控制台输出
      const color = colors[level];
      if (data !== undefined) {
        console.log(`${color}${formatted}${colors.reset}`, data);
      } else {
        console.log(`${color}${formatted}${colors.reset}`);
      }

      // 同时写入日志文件（仅在 CLI 模式启用）
      if (fileLoggingEnabled && logFilePath) {
        (async () => {
          try {
            const fs = await import('fs');
            const fileLog = data !== undefined
              ? `${formatted} ${JSON.stringify(data)}\n`
              : `${formatted}\n`;
            fs.appendFileSync(logFilePath!, fileLog, 'utf-8');
          } catch (error) {
            // 静默失败，不影响程序运行
          }
        })();
      }
    } else if (isBrowser) {
      // 浏览器环境：只输出到控制台
      const method = level === 'error' ? 'error' : level === 'warn' ? 'warn' : 'log';
      if (data !== undefined) {
        console[method](formatted, data);
      } else {
        console[method](formatted);
      }
    }
  }

  debug(message: string, data?: unknown): void {
    this.log('debug', `🔍 ${message}`, data);
  }

  info(message: string, data?: unknown): void {
    this.log('info', message, data);
  }

  warn(message: string, data?: unknown): void {
    this.log('warn', `⚠️ ${message}`, data);
  }

  error(message: string, data?: unknown): void {
    this.log('error', `❌ ${message}`, data);
  }
}

// 全局 logger 实例缓存
const loggers = new Map<string, Logger>();

/**
 * 获取或创建 Logger 实例
 * @param module 模块名称
 */
export function setupLogger(module: string): Logger {
  if (!loggers.has(module)) {
    loggers.set(module, new Logger(module));
  }
  return loggers.get(module)!;
}

/**
 * 设置所有 logger 的 debug 模式
 */
export function setGlobalDebug(enabled: boolean): void {
  loggers.forEach(logger => logger.setDebug(enabled));
}
