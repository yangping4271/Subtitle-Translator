# 字幕翻译工具 (Subtitle Translator)

一个集成了英文视频转录、字幕翻译和格式转换的命令行工具。将英文音频/视频转录为字幕，并翻译成多种语言，生成双语ASS字幕文件。

> ⚠️ **重要说明**：转录功能仅支持英文音频/视频。如果您的视频是其他语言，请先准备好英文SRT字幕文件。

## ✨ 核心特性

- 🎬 **英文视频转录**: 使用 Parakeet MLX 模型将英文音频/视频转录为SRT字幕
- 🌐 **智能翻译**: 支持多种LLM模型进行字幕翻译（中文、日文等）
- 📝 **双语字幕**: 自动生成双语ASS字幕文件
- ⚡ **批量处理**: 支持批量处理多个文件

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd Subtitle-Translator

# 使用 uv 安装
uv tool install .
```

### 配置

```bash
# 一键配置API密钥
subtitle-translate init
```

### 基本使用

```bash
# 🎯 最简单：批量处理当前目录所有文件（默认翻译成中文）
subtitle-translate

# 处理单个文件
subtitle-translate -i video.mp4

# 翻译成其他语言（如日语）
subtitle-translate -i video.mp4 -t ja

# 限制处理文件数量
subtitle-translate -n 3

# 启用反思翻译模式（提高质量）
subtitle-translate -i video.mp4 -r
```

**批量处理说明**：
- 自动扫描当前目录的 `.srt`, `.mp3`, `.mp4` 文件  
- 文件优先级：SRT > MP3 > MP4（避免重复处理）
- 跳过已存在 `.ass` 文件的项目

## 📖 详细说明

### 支持的输入格式

- **英文音频**: MP3（转录为字幕）
- **英文视频**: MP4, MOV, MKV等（转录为字幕）  
- **英文字幕**: SRT文件（直接翻译）

### 输出文件

- `文件名.srt` - 原始英文字幕
- `文件名.ass` - 双语ASS字幕文件

### 命令行参数

```
Usage: subtitle-translate [OPTIONS]

Options:
  -i, --input-file FILE    单个文件路径，不指定则批量处理当前目录
  -n, --count INTEGER      最大处理文件数量 [default: -1]
  -t, --target_lang TEXT   目标语言 [default: zh]
  -o, --output_dir PATH    输出目录 [default: 当前目录]
  -m, --llm-model TEXT     LLM模型
  -r, --reflect           启用反思翻译模式
  -d, --debug             调试模式
  --help                  显示帮助信息
```

## ⚙️ 配置

### 快速配置
```bash
subtitle-translate init
```

### 手动配置
创建 `.env` 文件：
```bash
# OpenAI API 配置（必需）
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
LLM_MODEL=gpt-4o-mini
```

## 🛠️ 开发

```bash
# 安装开发依赖
uv sync --dev

# 运行
uv run python -m subtitle_translator.cli --help
```

## 📄 许可证

MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。 