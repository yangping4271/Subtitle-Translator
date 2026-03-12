# 字幕翻译工具 (Subtitle Translator)

[English](./README.md) | [中文](./README_zh.md)

专注于多语言字幕翻译的命令行工具。将英文 SRT 字幕翻译成多种目标语言，并生成双语 ASS 字幕文件。

## 功能特性

- **多语言翻译**: 支持将英文字幕翻译成中文、日文、韩文、法文等多种目标语言
- **智能断句**: 多层降级策略，包括标点符号和基于规则的语义分割
- **AI驱动翻译**: 利用各种LLM模型实现高质量翻译
- **双语字幕**: 自动生成双语ASS字幕文件(英文+目标语言)
- **批量处理**: 同时处理多个文件
- **模块化配置**: 可为断句和翻译分别配置不同模型
- **外部上下文支持**: 通过 context.txt 文件提供额外的翻译上下文
- **术语表支持**: 支持全局和本地术语表，确保专业术语翻译一致性

## 快速开始

### 安装
```bash
git clone https://github.com/yangping4271/Subtitle-Translator.git
cd Subtitle-Translator
uv tool install .

# 更新 PATH 以使用已安装的工具
uv tool update-shell
# 然后重启终端或运行: source ~/.zshenv
```

### 配置

**推荐：使用交互式配置**

```bash
translate init
```

这将在 `~/.config/subtitle-translator/.env` 创建配置文件。

**方式二：直接设置环境变量**

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=your-api-key-here
export SPLIT_MODEL=gpt-4o-mini
export TRANSLATION_MODEL=gpt-4o
```

**方式三：手动创建配置文件**

创建 `~/.config/subtitle-translator/.env`：

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
SPLIT_MODEL=gpt-4o-mini
TRANSLATION_MODEL=gpt-4o
LLM_MODEL=gpt-4o-mini
```

### 基本使用
```bash
# 批量处理当前目录所有 SRT 文件（默认翻译成中文）
translate

# 处理单个文件
translate -i subtitle.srt

# 翻译到不同目标语言(中文、日文、韩文等)
translate -i subtitle.srt -t zh    # 中文(简体)
translate -i subtitle.srt -t ja    # 日文
translate -i subtitle.srt -t ko    # 韩文
translate -i subtitle.srt -t fr    # 法文

# 保留中间翻译文件
translate -i subtitle.srt -t zh --preserve-intermediate
```

## 工作流程

```
英文 SRT 字幕 → 智能断句 → AI 翻译 → 双语 ASS 字幕
```

## 支持的格式

### 输入格式
- **字幕文件**: `.srt` (英文字幕)

### 输出格式
- **英文 SRT**: 原始英文字幕
- **目标语言 SRT**: 翻译后的字幕
- **双语 ASS**: 英文+目标语言的双语字幕文件

## 使用外部上下文

在字幕文件同目录下创建 `context.txt` 或 `ctx.txt` 文件，提供额外的翻译上下文：

```bash
# 创建上下文文件
cat > context.txt << 'EOF'
This is a technical tutorial about Google Gemini CLI and AI agents.
Key topics: Gemini API, command-line tools, agent development.
Target audience: Developers and AI practitioners.
EOF

# 翻译时会自动读取并使用上下文
translate -i video.srt -t zh
```

上下文信息会与文件名和路径信息一起传递给翻译模型，帮助提高翻译质量和术语准确性。

### 术语表配置

项目支持自定义术语表，确保专业术语翻译的一致性。

**全局术语表**（用户配置目录）：
```bash
# 创建配置目录
mkdir -p ~/.config/subtitle-translator

# 编辑全局术语表
cat > ~/.config/subtitle-translator/terminology.txt << 'EOF'
# 全局术语表
[简体中文]
AGI = 通用人工智能 (AGI)
LLM = 大语言模型 (Large Language Model)
Transformer = Transformer

[繁体中文]
AGI = 通用人工智慧 (AGI)
EOF
```

**局部术语表**（字幕文件同目录）：
```bash
# 在字幕文件同目录创建 terminology.txt
# 局部术语会覆盖全局术语
cat > terminology.txt << 'EOF'
[简体中文]
# 覆盖全局术语
AGI = 人工通用智能 (AGI)
# 新增项目特定术语
project-term = 项目术语
EOF
```

**格式说明**：
- 支持 `#` 开头的注释行
- 使用 `[语言]` 标记语言段
- 使用 `术语 = 翻译` 格式
- 局部术语表会与全局术语表合并，相同术语以局部为准

## 命令行参考

### translate 命令
```bash
translate [OPTIONS]

选项:
  -i, --input-file FILE    单个文件路径。如不指定则批量处理当前目录
  -n, --count INTEGER      最大处理文件数量 [默认: -1]
  -t, --target-lang TEXT   目标语言 [默认: zh]
  -o, --output-dir PATH    输出目录 [默认: 当前目录]
  -m, --llm-model TEXT     LLM 模型
  --split-model TEXT       断句模型
  --translation-model TEXT 翻译模型
  -p, --preserve-intermediate  保留中间翻译文件
  --dry-run                预览模式，不实际执行
```

### 支持的语言

**源语言:**
- 英文 (English)

**目标语言:**
支持将英文翻译成多种语言：
- **中文**: `zh` (简体), `zh-cn` (简体), `zh-tw` (繁体)
- **亚洲语言**: `ja` (日文), `ko` (韩文), `th` (泰文), `vi` (越南文)
- **欧洲语言**: `fr` (法文), `de` (德文), `es` (西班牙文), `pt` (葡萄牙文), `it` (意大利文), `ru` (俄文)
- **其他**: `ar` (阿拉伯文) 等

> **注意**: 本工具专注于英文字幕的翻译，需要提供英文 SRT 字幕文件作为输入。

## 配置

**推荐：使用交互式配置**

```bash
translate init
```

这将在 `~/.config/subtitle-translator/.env` 创建配置文件。

**方式二：直接设置环境变量**

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=your-api-key-here
export SPLIT_MODEL=gpt-4o-mini
export TRANSLATION_MODEL=gpt-4o
```

**方式三：手动创建配置文件**

创建 `~/.config/subtitle-translator/.env`：

```bash
# OpenAI API 配置（必需）
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here

# 模型配置
SPLIT_MODEL=gpt-4o-mini      # 断句模型
TRANSLATION_MODEL=gpt-4o     # 翻译模型
LLM_MODEL=gpt-4o-mini        # 默认模型
```

## 开发

```bash
# 安装开发依赖
uv sync --dev

# 运行主程序
uv run python -m subtitle_translator.cli --help
```


## 许可证

MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Video Captioner](https://github.com/WEIFENG2333/VideoCaptioner) - 智能字幕助手项目
- [uv](https://github.com/astral-sh/uv) - 现代化的 Python 包管理工具
- [Typer](https://github.com/tiangolo/typer) - 出色的命令行接口框架

---

**📧 联系方式**: 如有问题或建议，请通过 Issues 或 Pull Requests 联系我们。 