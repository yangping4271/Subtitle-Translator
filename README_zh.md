# 字幕翻译工具

[English](./README.md) | [中文](./README_zh.md)

一个把英文 `.srt` 字幕翻译成其他语言，并生成双语 `.ass` 字幕的命令行工具。

## 主要能力

- 把英文字幕翻译成中文、日文、韩文、法文等多种语言
- 输出翻译后的 `.srt` 和双语 `.ass`
- 支持 OpenAI-compatible API，也支持 LM Studio 这类本地服务
- 支持通过 `context.txt` / `ctx.txt` 提供额外上下文
- 支持全局和局部术语表，保持术语翻译一致

## 安装

```bash
git clone https://github.com/yangping4271/Subtitle-Translator.git
cd Subtitle-Translator
uv tool install .
uv tool update-shell
```

然后重启终端，或者执行 `source ~/.zshenv`。

## 快速开始

先初始化配置：

```bash
translate init
```

开始翻译：

```bash
# 把当前目录里的所有 SRT 文件翻译成中文
translate

# 处理单个文件
translate -i subtitle.srt

# 指定目标语言
translate -i subtitle.srt -t ja
translate -i subtitle.srt -t fr

# 保留中间文件
translate -i subtitle.srt -t zh --preserve-intermediate
```

源语言只支持英文。常用目标语言代码包括 `zh`、`zh-tw`、`ja`、`ko`、`fr`、`de`、`es`、`pt`、`it`、`ru`、`ar`、`th`、`vi`。

## 配置

推荐直接运行：

```bash
translate init
```

它会创建 `~/.config/subtitle-translator/.env`。

也可以手动配置：

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
SPLIT_MODEL=your-split-model
TRANSLATION_MODEL=your-translation-model
LLM_MODEL=your-default-model
```

如果你使用 LM Studio 或其他本地 OpenAI-compatible 服务：

```bash
OPENAI_BASE_URL=http://127.0.0.1:1234/v1
OPENAI_API_KEY=
SPLIT_MODEL=your-local-split-model
TRANSLATION_MODEL=your-local-translation-model
LLM_MODEL=your-local-default-model
```

## 可选：上下文与术语表

如果需要额外上下文，把 `context.txt` 或 `ctx.txt` 放到字幕文件同目录即可。

如果需要固定术语翻译，可以使用术语表：

- 全局术语表：`~/.config/subtitle-translator/terminology.txt`
- 局部覆盖：字幕文件目录下的 `terminology.txt`

基本格式：

```text
[简体中文]
LLM = 大语言模型 (Large Language Model)
AGI = 通用人工智能 (AGI)
LangChain = LangChain | aliases: land chain, lang chain
```

`aliases` 用于提示模型修正常见语音识别错误。上例会把 `land chain`、`lang chain` 作为可能的 ASR 错误，优先纠正为 `LangChain` 后再翻译。

局部术语表会在全局术语表之上覆盖同名术语。

默认还会为简体中文启用外部术语库缓存，领域为 `programming,tech,education`。外部术语不会全量注入 prompt，只会在当前字幕批次命中时动态加入。可通过环境变量调整：

```bash
EXTERNAL_GLOSSARY_ENABLED=true
EXTERNAL_GLOSSARY_DOMAINS=programming,tech,education
EXTERNAL_GLOSSARY_MAX_TERMS=40
```

## CLI

完整参数请看 `translate --help`。

## Codex Skill

这个仓库也内置了一个 Codex skill，目录在 [`subtitle-translator/`](./subtitle-translator/)。

安装到本地 Codex skills 目录：

```bash
mkdir -p ~/.codex/skills
cp -R subtitle-translator ~/.codex/skills/subtitle-translator
```

然后重启 Codex。重启后可以通过 `$subtitle-translator` 调用。

这个 skill 可以：

- 在需要时发现或安装 `translate` CLI
- 翻译单个 `.srt` 文件或整个目录
- 更新 `terminology.txt` 术语和 ASR `aliases`
- 在重跑翻译前补充或修正 `context.txt` / `ctx.txt`

## 开发

```bash
uv sync --dev
uv run python -m subtitle_translator.cli --help
```

## 许可证

MIT，详见 [LICENSE](LICENSE)。
