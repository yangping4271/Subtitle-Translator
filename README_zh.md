# 字幕翻译工具 (Subtitle Translator)

[English](./README.md) | [中文](./README_zh.md)

集成了英文视频转录、字幕翻译的命令行工具。将英文音频/视频转录为字幕，并翻译成多种语言，生成双语ASS字幕文件。

> ⚠️ **重要**：转录功能仅支持英文音频/视频。如果您的视频是其他语言，请先准备好英文SRT字幕文件。

## 功能特性

- **英文视频转录**: 使用 Parakeet MLX 模型将英文音频/视频转录为SRT字幕
- **智能翻译**: 支持多种LLM模型，支持多种语言翻译
- **文件路径上下文智能**: 利用文件名和文件夹名称提升翻译准确性和术语一致性
- **双语字幕**: 自动生成双语ASS字幕文件
- **批量处理**: 支持批量处理多个文件
- **模块化配置**: 支持为断句、翻译、总结分别配置不同模型

## 快速开始

### 安装
```bash
git clone <your-repo-url>
cd Subtitle-Translator
uv tool install .

# 更新 PATH 以使用已安装的工具
uv tool update-shell
# 然后重启终端或运行: source ~/.zshenv
```

### 配置
```bash
translate init  # 一键配置API密钥
```

### 基本使用
```bash
# 批量处理当前目录所有文件（默认翻译成中文）
translate

# 处理单个文件
translate -i video.mp4

# 翻译成其他语言
translate -i video.mp4 -t ja

# 启用反思翻译模式（提高质量）
translate -i video.mp4 -r

# 仅转录音频/视频（不翻译）
transcribe video.mp4

# 转录多个文件
transcribe audio1.mp3 audio2.wav video.mp4

# 生成词级别时间戳
transcribe video.mp4 --timestamps

# 输出多种格式
transcribe video.mp4 --output-format all
```

## 工作流程

### 完整流程 (translate 命令)
```
音频/视频 → 转录 → 英文SRT → 翻译 → 双语ASS字幕
```

### 仅翻译流程 (已有英文字幕)
```
英文SRT → 翻译 → 双语ASS字幕
```

### 仅转录流程 (transcribe 命令)
```
音频/视频 → 转录 → 多种格式输出
```

## 支持的格式

### 输入格式
- **音频**: MP3, WAV, FLAC, M4A, AAC 等
- **视频**: MP4, MOV, MKV, AVI, WebM 等
- **字幕**: SRT 格式

### 输出格式
- **translate**: 生成 `.srt` (英文) 和 `.ass` (双语) 文件
- **transcribe**: 支持 TXT、SRT、VTT、JSON 等多种格式

## 转录功能特性

基于 Parakeet MLX 模型的专业转录工具：

- **高性能**: 基于 Apple MLX 框架，在 Apple Silicon 上性能卓越
- **智能分块**: 自动处理长音频文件，避免内存溢出
- **精确时间戳**: 支持词级别时间戳，精确到毫秒
- **批量处理**: 一次转录多个音频文件

### 高级用法
```bash
# 处理长音频（自动分块）
transcribe long_podcast.mp3 --chunk-duration 120 --overlap-duration 15

# 自定义输出目录和文件名
transcribe interview.mp3 --output-dir ./transcripts --output-template "interview_{filename}"

# 高精度模式
transcribe audio.mp3 --fp32
```

## 命令行参考

### translate 命令
```bash
translate [OPTIONS] [COMMAND]

Options:
  -i, --input-file FILE    单个文件路径，不指定则批量处理当前目录
  -n, --count INTEGER      最大处理文件数量 [default: -1]
  -t, --target_lang TEXT   目标语言 [default: zh]
  -o, --output_dir PATH    输出目录 [default: 当前目录]
  --model TEXT             转录模型
  -m, --llm-model TEXT     LLM模型
  -r, --reflect            启用反思翻译模式
  -d, --debug              调试模式
  
Commands:
  init                     初始化配置
```

### transcribe 命令
```bash
transcribe [OPTIONS] AUDIOS...

Options:
  --model TEXT                    转录模型 [default: mlx-community/parakeet-tdt-0.6b-v2]
  --output-dir PATH               输出目录 [default: .]
  --output-format [txt|srt|vtt|json|all]  输出格式 [default: srt]
  --output-template TEXT          文件名模板 [default: {filename}]
  --timestamps/--no-timestamps    输出词级别时间戳 [default: False]
  --chunk-duration FLOAT          分块时长（秒）[default: 120.0]
  --overlap-duration FLOAT        重叠时长（秒）[default: 15.0]
  -v, --verbose                   显示详细信息
  --fp32/--bf16                   使用FP32精度 [default: bf16]
```

### 支持的翻译语言
支持多种语言翻译，常用语言代码：`zh`（中文）、`ja`（日文）、`ko`（韩文）、`en`（英文）、`fr`（法文）等。

## 配置

### 快速配置
```bash
translate init
```

交互式配置包括：
- LLM 服务的 API 密钥设置
- 不同任务的模型配置
- **Hugging Face 镜像站配置**（提高模型下载速度）

### 手动配置
创建 `.env` 文件：
```bash
# OpenAI API 配置（必需）
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here

# Hugging Face 镜像站配置（可选，提高下载速度）
# 推荐国内用户或与 huggingface.co 连接较慢的用户使用
HF_ENDPOINT=https://hf-mirror.com

# 模型配置
SPLIT_MODEL=gpt-4o-mini      # 断句模型
TRANSLATION_MODEL=gpt-4o     # 翻译模型
SUMMARY_MODEL=gpt-4o-mini    # 总结模型
LLM_MODEL=gpt-4o-mini        # 默认模型
```

### Hugging Face 镜像站配置

为了提高模型下载的可靠性和速度，特别是对国内用户，可以配置 Hugging Face 镜像站：

#### 方式一：交互式配置
```bash
translate init
# 在提示是否使用 Hugging Face 镜像站时选择 "是"
```

#### 方式二：环境变量
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

#### 方式三：添加到 .env 文件
```bash
# 在你的 .env 文件中添加这一行
HF_ENDPOINT=https://hf-mirror.com
```

#### 支持的镜像站
- **hf-mirror.com**（推荐国内用户使用）
- **huggingface.co**（官方地址，默认）
- 自定义镜像站地址

系统会自动检测网络连通性并选择最佳下载方式：
1. **huggingface-cli** + 配置的镜像站（如可用）
2. **hf_hub_download** + 配置的镜像站
3. **自动故障转移**：主镜像站失败时自动切换到备用镜像站

## 开发

```bash
# 安装开发依赖
uv sync --dev

# 运行主程序
uv run python -m subtitle_translator.cli --help

# 运行转录功能
uv run python -m subtitle_translator.transcription_core.cli --help
```

## 许可证

MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Parakeet MLX](https://github.com/senstella/parakeet-mlx) - Nvidia Parakeet 模型在 Apple Silicon 上使用 MLX 的实现
- [Video Captioner](https://github.com/WEIFENG2333/VideoCaptioner) - 智能字幕助手项目
- [uv](https://github.com/astral-sh/uv) - 现代化的 Python 包管理工具
- [Typer](https://github.com/tiangolo/typer) - 出色的命令行接口框架

---

**📧 联系方式**: 如有问题或建议，请通过 Issues 或 Pull Requests 联系我们。 