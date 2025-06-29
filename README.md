# 字幕翻译工具 (Subtitle Translator)

一个集成了视频转录、字幕翻译和格式转换的强大命令行工具。支持将视频文件转录为字幕，并将字幕翻译成多种语言，最终生成双语ASS字幕文件。

## ✨ 核心特性

- 🎬 **视频转录**: 使用先进的 Parakeet MLX 模型将视频转录为高质量 SRT 字幕
- 🌐 **智能翻译**: 支持多种 LLM 模型（GPT、Claude、Gemini等）进行字幕翻译
- 📝 **格式转换**: 自动生成双语 ASS 字幕文件，支持中英文对照显示
- ⚡ **并行处理**: 多线程处理，大幅提升翻译效率
- 🎯 **批量处理**: 支持单文件和批量处理模式
- 🔧 **智能配置**: 跨目录环境变量加载，支持全局和项目级配置
- 🧹 **自动清理**: 处理完成后自动清理临时文件

## 🚀 快速开始

### 安装

确保你的系统已安装 Python 3.8+ 和 [uv](https://github.com/astral-sh/uv)。

```bash
# 克隆仓库
git clone <your-repo-url>
cd subtitle_translator

# 使用 uv 安装
uv tool install .
```

### 配置

创建 `.env` 文件配置 API 密钥：

```bash
# 在项目根目录或 ~/.config/subtitle_translator/ 目录下创建 .env 文件
cat > .env << EOF
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
LLM_MODEL=gpt-4o-mini
EOF
```

### 基本使用

```bash
# 处理单个视频文件
subtitle-translate -i video.mp4 -t zh

# 批量处理当前目录的所有视频
subtitle-translate -t zh

# 指定输出目录
subtitle-translate -i video.mp4 -t zh -o ./output

# 启用反思翻译模式（提高质量）
subtitle-translate -i video.mp4 -t zh -r

# 调试模式
subtitle-translate -i video.mp4 -t zh -d
```

## 📖 详细使用说明

### 命令行参数

```
Usage: subtitle-translate [OPTIONS]

Options:
  -i, --input-file FILE         要处理的单个文件路径
  -n, --count INTEGER          最大处理文件数量，-1表示处理所有文件 [default: -1]
  -t, --target_lang TEXT       目标翻译语言 [default: zh]
  -o, --output_dir PATH        输出文件的目录 [default: 当前目录]
  --model TEXT                 用于转录的 Parakeet MLX 模型
  -m, --llm-model TEXT         用于翻译的LLM模型
  -r, --reflect                启用反思翻译模式
  -d, --debug                  启用调试日志级别
  --help                       显示此帮助信息
```

### 支持的输入格式

- **视频格式**: MP4, AVI, MOV, MKV 等（通过 Parakeet MLX 转录）
- **字幕格式**: SRT 文件（直接翻译）

### 输出文件

- `原文件名.srt` - 原始转录字幕（如果是视频输入）
- `原文件名.zh.srt` - 中文翻译字幕
- `原文件名.en.srt` - 英文优化字幕
- `原文件名.ass` - 双语ASS字幕文件（最终输出）

## ⚙️ 配置说明

### 环境变量

工具支持灵活的配置方式，按优先级顺序：

1. **项目配置** (优先级最高): 项目根目录的 `.env` 文件
2. **全局配置**: `~/.config/subtitle_translator/.env` 文件
3. **系统环境变量**: 系统级环境变量

### 必需配置

```bash
# OpenAI API 配置
OPENAI_BASE_URL=https://api.openai.com/v1  # API 端点
OPENAI_API_KEY=your-api-key-here           # API 密钥
LLM_MODEL=gpt-4o-mini                      # 默认模型
```

### 可选配置

```bash
# 日志级别
LOG_LEVEL=INFO

# 调试模式
DEBUG=false
```

### 支持的 LLM 模型

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-sonnet`, `claude-3-haiku`
- **Google**: `google/gemini-2.5-flash-lite-preview-06-17`
- 更多模型请参考你的 API 提供商文档

## 🏗️ 技术架构

### 核心模块

```
subtitle_translator/
├── cli.py                    # 命令行接口
├── transcription_core/       # 转录核心
│   ├── parakeet.py          # Parakeet MLX 集成
│   ├── audio.py             # 音频处理
│   └── ...
└── translation_core/        # 翻译核心
    ├── optimizer.py         # 字幕翻译优化
    ├── summarizer.py        # 内容摘要
    ├── spliter.py           # 智能断句
    └── utils/
        ├── srt2ass.py       # SRT 到 ASS 转换
        └── ...
```

### 处理流程

1. **音频提取** - 从视频文件提取音频
2. **语音转录** - 使用 Parakeet MLX 模型转录为文本
3. **智能断句** - 使用 LLM 进行语义断句
4. **内容摘要** - 生成字幕内容摘要作为翻译上下文
5. **并行翻译** - 多线程并行翻译字幕片段
6. **质量优化** - 可选的反思翻译模式
7. **格式转换** - 生成双语 ASS 字幕文件

### 性能优化

- **多线程处理**: 默认 18 个线程并行翻译
- **批量请求**: 支持批量 API 调用
- **智能缓存**: 避免重复处理
- **内存管理**: 自动清理临时文件

## 🛠️ 开发指南

### 本地开发

```bash
# 克隆仓库
git clone <your-repo-url>
cd subtitle_translator

# 安装开发依赖
uv sync --dev

# 运行测试
uv run python -m subtitle_translator.cli --help
```

### 依赖管理

项目使用 `uv` 进行依赖管理：

- `pyproject.toml` - 项目配置和依赖声明
- `uv.lock` - 锁定的依赖版本

### 关键依赖

- `parakeet-mlx` - 语音转录
- `python-dotenv` - 环境变量管理
- `typer` - 命令行接口
- `rich` - 富文本输出
- `openai` - LLM API 调用

## ⚠️ 已知问题

### 1. 环境变量重复加载输出
- **现象**: 运行时出现多次"已加载环境配置"消息
- **影响**: 仅输出显示，不影响功能
- **状态**: 正在调查中

### 2. SRT2ASS 语法警告
- **现象**: 正则表达式转义序列警告
- **影响**: 不影响功能，仅显示警告
- **状态**: 计划修复

## 📈 使用统计

经过实际测试验证：

- ✅ 成功处理 96 个字幕条目的视频文件
- ✅ 生成 20KB 的高质量双语 ASS 文件
- ✅ 翻译准确性和术语一致性优秀
- ✅ 多线程处理稳定高效

## 🤝 贡献指南

欢迎贡献代码和反馈问题！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Parakeet MLX](https://github.com/parakeet-ai/parakeet) - 优秀的语音转录模型
- [uv](https://github.com/astral-sh/uv) - 现代化的 Python 包管理工具
- [Typer](https://github.com/tiangolo/typer) - 出色的命令行接口框架

---

**📧 联系方式**: 如有问题或建议，请通过 Issues 或 Pull Requests 联系我们。 