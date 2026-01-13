/**
 * CLI 工具 - 本地调试入口
 * 提供与 Python 版本一致的命令行接口和日志输出
 */

import { Command } from 'commander';
import { config as dotenvConfig } from 'dotenv';
import { existsSync, readFileSync, writeFileSync } from 'fs';
import { dirname, join, resolve } from 'path';
import { loadConfig, validateConfig } from '../utils/config.js';
import { createTranslatorService } from '../services/translator-service.js';
import { setGlobalDebug, setupLogger, initFileLogging } from '../utils/logger.js';
import { SUPPORTED_LANGUAGES, getLanguageName } from '../utils/language.js';
import type { SubtitleEntry } from '../types/index.js';

// 在 CLI 入口加载 .env 文件
const envPaths = [
  join(process.cwd(), '.env'),
  join(process.cwd(), '..', '.env'),
];
for (const envPath of envPaths) {
  if (existsSync(envPath)) {
    dotenvConfig({ path: envPath });
    break;
  }
}

// 初始化文件日志（CLI 模式）
const logDir = join(process.cwd(), 'log');
await initFileLogging(logDir);

const logger = setupLogger('cli');

/**
 * 解析 SRT 文件内容
 */
function parseSRT(content: string): SubtitleEntry[] {
  const entries: SubtitleEntry[] = [];
  const blocks = content.trim().split(/\n\n+/);

  for (const block of blocks) {
    const lines = block.split('\n');
    if (lines.length < 3) continue;

    // 解析索引
    const index = parseInt(lines[0], 10);
    if (isNaN(index)) continue;

    // 解析时间戳
    const timeMatch = lines[1].match(
      /(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})/
    );
    if (!timeMatch) continue;

    const startTime =
      parseInt(timeMatch[1], 10) * 3600000 +
      parseInt(timeMatch[2], 10) * 60000 +
      parseInt(timeMatch[3], 10) * 1000 +
      parseInt(timeMatch[4], 10);

    const endTime =
      parseInt(timeMatch[5], 10) * 3600000 +
      parseInt(timeMatch[6], 10) * 60000 +
      parseInt(timeMatch[7], 10) * 1000 +
      parseInt(timeMatch[8], 10);

    // 解析文本
    const text = lines.slice(2).join(' ').trim();

    entries.push({ index, startTime, endTime, text });
  }

  return entries;
}

/**
 * 格式化时间戳为 ASS 格式
 */
function formatASSTime(ms: number): string {
  const hours = Math.floor(ms / 3600000);
  const minutes = Math.floor((ms % 3600000) / 60000);
  const seconds = Math.floor((ms % 60000) / 1000);
  const centiseconds = Math.floor((ms % 1000) / 10);

  return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(centiseconds).padStart(2, '0')}`;
}

/**
 * 语言到字体的映射（与 Python ass_converter.py 保持一致）
 */
const LANGUAGE_FONTS: Record<string, string> = {
  'zh': '宋体-简 黑体,11',
  'zh-cn': '宋体-简 黑体,11',
  'zh-tw': 'Noto Sans CJK TC,12',
  'ja': 'Noto Sans CJK JP,13',
  'ko': 'Noto Sans CJK KR,12',
  'fr': 'Noto Sans,14',
  'de': 'Noto Sans,14',
  'es': 'Noto Sans,14',
  'pt': 'Noto Sans,14',
  'ru': 'Noto Sans,13',
  'it': 'Noto Sans,14',
  'ar': 'Noto Sans Arabic,13',
  'th': 'Noto Sans Thai,13',
  'vi': 'Noto Sans,13',
  'default': 'Noto Sans,13',
};

/**
 * 生成 ASS 字幕文件内容（与 Python ass_converter.py 保持一致）
 */
function generateASS(
  english: SubtitleEntry[],
  chinese: SubtitleEntry[],
  targetLang: string
): string {
  // 获取目标语言字体配置
  const targetFont = LANGUAGE_FONTS[targetLang] || LANGUAGE_FONTS['default'];

  // ASS 文件头（与 Python 版本完全一致）
  let ass = `[Script Info]
; This is an Advanced Sub Station Alpha v4+ script.
Title:
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Serif,18,&H0000FFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,1,1,7,1
Style: Secondary,${targetFont},&H0000FF00,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,1,1,7,1

[Events]
Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text
`;

  // 添加目标语言字幕（Secondary 样式，显示在下方，绿色）
  for (const entry of chinese) {
    const start = formatASSTime(entry.startTime);
    const end = formatASSTime(entry.endTime);
    ass += `Dialogue: 0,${start},${end},Secondary,,0,0,0,,${entry.text}\n`;
  }

  // 添加英文字幕（Default 样式，显示在上方，青色）
  for (const entry of english) {
    const start = formatASSTime(entry.startTime);
    const end = formatASSTime(entry.endTime);
    ass += `Dialogue: 0,${start},${end},Default,,0,0,0,,${entry.text}\n`;
  }

  return ass;
}

// 创建 CLI 程序
const program = new Command();

program
  .name('subtitle-translate')
  .description('字幕翻译 CLI 工具（TypeScript 版本，与 Python 版本一致）')
  .version('1.0.0');

program
  .command('translate')
  .description('翻译 SRT 字幕文件')
  .requiredOption('-i, --input <file>', '输入 SRT 文件路径')
  .option('-o, --output <file>', '输出文件路径（默认为输入文件名.ass）')
  .option('-t, --target <lang>', '目标语言代码', 'zh')
  .option('-d, --debug', '调试模式')
  .action(async (options) => {
    try {
      // 设置调试模式
      if (options.debug) {
        setGlobalDebug(true);
        logger.info('🔍 调试模式已启用');
      }

      // 加载配置
      logger.info('📋 加载配置...');
      const config = await loadConfig();

      // 验证配置
      const errors = validateConfig(config);
      if (errors.length > 0) {
        for (const error of errors) {
          logger.error(error);
        }
        process.exit(1);
      }

      // 更新目标语言
      config.targetLanguage = options.target;

      logger.info(`🌐 目标语言: ${getLanguageName(options.target)}`);
      logger.info(`📝 输入文件: ${options.input}`);

      // 读取输入文件
      if (!existsSync(options.input)) {
        logger.error(`文件不存在: ${options.input}`);
        process.exit(1);
      }

      const content = readFileSync(options.input, 'utf-8');
      const subtitles = parseSRT(content);

      if (subtitles.length === 0) {
        logger.error('未找到有效的字幕条目');
        process.exit(1);
      }

      logger.info(`📊 共 ${subtitles.length} 条字幕`);

      // 创建翻译服务
      const service = createTranslatorService(config);

      // 执行翻译
      logger.info('🚀 开始翻译...');
      const startTime = Date.now();

      const result = await service.translateFull(subtitles, {
        inputFile: options.input,
        onProgress: (step, current, total) => {
          const stepNames: Record<string, string> = {
            split: '断句优化',
            summary: '内容总结',
            translate: '翻译字幕',
            complete: '完成',
          };
          const stepName = stepNames[step] || step;
          const progress = Math.round((current / total) * 100);
          logger.info(`📈 ${stepName}: ${progress}%`);
        },
      });

      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      logger.info(`⏱️ 翻译耗时: ${duration}秒`);

      // 生成输出文件（与输入文件在同一目录）
      const inputAbsPath = resolve(options.input);
      const outputPath = options.output ||
        inputAbsPath.replace(/\.[^/.]+$/, '.ass');

      const assContent = generateASS(result.english, result.chinese, options.target);
      writeFileSync(outputPath, assContent, 'utf-8');

      logger.info(`✅ 翻译完成: ${outputPath}`);

    } catch (error) {
      logger.error(`翻译失败: ${error}`);
      process.exit(1);
    }
  });

program
  .command('test-api')
  .description('测试 API 连接')
  .action(async () => {
    try {
      logger.info('🔍 测试 API 连接...');

      const config = await loadConfig();
      const errors = validateConfig(config);

      if (errors.length > 0) {
        for (const error of errors) {
          logger.error(error);
        }
        process.exit(1);
      }

      logger.info(`📡 API 地址: ${config.openaiBaseUrl}`);
      logger.info(`🤖 翻译模型: ${config.translationModel}`);

      // 简单测试
      const response = await fetch(`${config.openaiBaseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${config.openaiApiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: config.translationModel,
          messages: [
            { role: 'user', content: 'Say "API test successful" in one line.' },
          ],
          max_tokens: 20,
        }),
      });

      if (response.ok) {
        const data = await response.json() as {
          choices?: Array<{ message?: { content?: string } }>;
        };
        logger.info(`✅ API 连接成功: ${data.choices?.[0]?.message?.content}`);
      } else {
        const error = await response.json() as { error?: { message?: string } };
        logger.error(`❌ API 连接失败: ${error.error?.message || response.status}`);
        process.exit(1);
      }

    } catch (error) {
      logger.error(`❌ 测试失败: ${error}`);
      process.exit(1);
    }
  });

program
  .command('languages')
  .description('列出支持的目标语言')
  .action(() => {
    logger.info('📋 支持的目标语言:');
    for (const lang of SUPPORTED_LANGUAGES) {
      logger.info(`   ${lang} -> ${getLanguageName(lang)}`);
    }
  });

// 解析命令行参数
program.parse();
