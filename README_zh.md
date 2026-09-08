# 字幕翻译工具

[English](./README.md) | [中文](./README_zh.md)

一个把英文 `.srt` 字幕翻译成其他语言，并生成双语 `.ass` 字幕的命令行工具。

## 主要能力

- 把英文字幕翻译成中文、日文、韩文、法文等多种语言
- 输出双语 `.ass`，并可选择保留中间 `.srt` 文件
- 支持远程 OpenAI-compatible API
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

默认会在当前目录按输入文件名生成 `.ass` 格式的双语结果；使用 `--input-dir` 时会输出到输入目录。可以通过 `-o OUTPUT_DIR` 指定其他输出目录。

## 配置

推荐直接运行：

```bash
translate init
```

它会创建 `~/.config/subtitle-translator/.env`。

项目只支持远程 API 端点，且 `OPENAI_API_KEY` 为必填项；已有的本地服务或无鉴权端点配置需要替换为带认证的远程 API。

模型名没有内置默认值。必须设置 `SPLIT_MODEL` 和 `TRANSLATION_MODEL`；也可以只设 `LLM_MODEL`，同时用于断句和翻译。

也可以手动配置：

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
SPLIT_MODEL=your-split-model
TRANSLATION_MODEL=your-translation-model
LOG_RAW_PAYLOADS=false
```

日志固定保存在 `~/.local/share/subtitle-translator/logs/app.log`，会自动轮转并使用私有文件权限。默认不记录完整字幕请求和模型原始响应；只有排查 payload 问题时才建议临时设置 `LOG_RAW_PAYLOADS=true`。

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
EXTERNAL_GLOSSARY_MAX_TERMS=12
```

## CLI

完整参数请看 `translate --help`。

## 开发

```bash
uv sync --dev
uv run python -m subtitle_translator.cli --help
```

## 许可证

MIT，详见 [LICENSE](LICENSE)。
