import * as esbuild from 'esbuild';
import * as fs from 'fs';
import * as path from 'path';

const isProduction = process.env.NODE_ENV === 'production';
const isWatch = process.argv.includes('--watch');

// 通用浏览器配置
const browserConfig = {
  bundle: true,
  platform: 'browser',
  target: 'es2020',
  format: 'iife',
  minify: isProduction,
  sourcemap: !isProduction,
  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
    '__PLATFORM__': '"browser"',
  },
};

// 扩展各入口文件配置
const extensionEntries = [
  {
    entryPoints: ['src/extension/translator.ts'],
    outfile: 'extension/translator.js',
    globalName: 'TranslatorModule',
  },
  {
    entryPoints: ['src/extension/config.ts'],
    outfile: 'extension/config.js',
    globalName: 'ConfigModule',
  },
  {
    entryPoints: ['src/extension/inject.ts'],
    outfile: 'extension/inject.js',
  },
  {
    entryPoints: ['src/extension/subtitle-parser.ts'],
    outfile: 'extension/subtitle-parser.js',
    globalName: 'SubtitleParserModule',
  },
  {
    entryPoints: ['src/extension/transcript-core.ts'],
    outfile: 'extension/transcript-core.js',
  },
  {
    entryPoints: ['src/extension/background.ts'],
    outfile: 'extension/background.js',
    globalName: 'BackgroundModule',
  },
  {
    entryPoints: ['src/extension/content.ts'],
    outfile: 'extension/content.js',
    globalName: 'ContentModule',
  },
];

// CLI 构建配置
const cliConfig = {
  entryPoints: ['src/cli/index.ts'],
  bundle: true,
  outfile: 'dist/cli.js',
  platform: 'node',
  target: 'node18',
  format: 'esm',
  minify: false,
  sourcemap: true,
  define: {
    '__PLATFORM__': '"node"',
  },
  banner: {
    js: '#!/usr/bin/env node',
  },
  external: ['commander', 'dotenv'],
};

// 复制静态文件到 extension 目录
function copyStaticFiles() {
  // 只复制真正的静态文件（非 TS 编译的）
  const staticFiles = [
    'manifest.json',
    'popup.html',
    'popup.js',     // 暂时保留 popup.js（待迁移到 TS）
    'popup.css',
    'subtitle-overlay.css',
    'transcript-styles.css',
  ];

  // 确保 extension 目录存在
  if (!fs.existsSync('extension')) {
    fs.mkdirSync('extension', { recursive: true });
  }

  // 复制文件
  staticFiles.forEach(file => {
    if (fs.existsSync(file)) {
      fs.copyFileSync(file, path.join('extension', file));
    }
  });

  // 复制 icons 目录
  if (fs.existsSync('icons')) {
    const iconsDir = path.join('extension', 'icons');
    if (!fs.existsSync(iconsDir)) {
      fs.mkdirSync(iconsDir, { recursive: true });
    }
    fs.readdirSync('icons').forEach(file => {
      fs.copyFileSync(
        path.join('icons', file),
        path.join(iconsDir, file)
      );
    });
  }

  console.log('📁 Static files copied to extension/');
}

async function build() {
  try {
    // 确保输出目录存在
    if (!fs.existsSync('extension')) {
      fs.mkdirSync('extension', { recursive: true });
    }
    if (!fs.existsSync('dist')) {
      fs.mkdirSync('dist', { recursive: true });
    }

    // 复制静态文件
    copyStaticFiles();

    // 构建所有扩展入口
    const extensionConfigs = extensionEntries.map(entry => ({
      ...browserConfig,
      ...entry,
    }));

    if (isWatch) {
      // 开发模式：监听文件变化
      const contexts = await Promise.all([
        ...extensionConfigs.map(config => esbuild.context(config)),
        esbuild.context(cliConfig),
      ]);

      await Promise.all(contexts.map(ctx => ctx.watch()));
      console.log('👀 Watching for changes...');
    } else {
      // 一次性构建
      console.log('🔨 Building...');

      await Promise.all([
        ...extensionConfigs.map(config => esbuild.build(config)),
        esbuild.build(cliConfig),
      ]);

      console.log('✅ Build complete');
      console.log('   📦 Extension files:');
      extensionEntries.forEach(entry => {
        console.log(`      - ${entry.outfile}`);
      });
      console.log('   📦 dist/cli.js (CLI tool)');
    }
  } catch (error) {
    console.error('❌ Build failed:', error);
    process.exit(1);
  }
}

build();
