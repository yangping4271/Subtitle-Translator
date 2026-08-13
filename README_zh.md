# 字幕翻译工具

[English](./README.md) | [中文](./README_zh.md)

一个把英文 `.srt` 字幕翻译成其他语言，并生成双语 `.ass` 字幕的命令行工具。

## 主要能力

- 把英文字幕翻译成中文、日文、韩文、法文等多种语言
- 输出翻译后的 `.srt` 和双语 `.ass`
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

## 配置

推荐直接运行：

```bash
translate init
```

它会创建 `~/.config/subtitle-translator/.env`。

项目只支持远程 API 端点，且 `OPENAI_API_KEY` 为必填项；已有的本地服务或无鉴权端点配置需要替换为带认证的远程 API。

也可以手动配置：

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
SPLIT_MODEL=your-split-model
TRANSLATION_MODEL=your-translation-model
LLM_MODEL=your-default-model
DISABLE_THINKING=true
LOG_RAW_PAYLOADS=false
```

`DISABLE_THINKING` 默认为 `true`。额外的推理/思考 token 按两种方式关闭：

- 能识别官方网址、且该供应商有统一关闭参数时，对所有模型关闭：
  - OpenRouter：`reasoning.effort=none`
  - DeepSeek、智谱、MiniMax：`thinking.type=disabled`
- 其他端点（包括 OpenAI / Kimi / Google / Groq 官方）只对已登记的模型名关闭：
  - `gpt-5.6`、`gpt-5.6-sol`、`gpt-5.6-terra`、`gpt-5.6-luna`、`gpt-oss-120b`、`gpt-oss-20b`、`qwen3-32b` 发送 `reasoning_effort=none`
  - `deepseek-v4-flash`、`deepseek-v4-pro`、`glm-5.2`、`glm-5.1`、`glm-5`、`glm-5-turbo`、`glm-4.7`、`glm-4.6`、`glm-4.5`、`kimi-k2.6`、`kimi-k2.5`、`MiniMax-M3` 发送 `thinking.type=disabled`
  - `gemini-3.6-flash`、`gemini-3.5-flash`、`gemini-3-flash-preview` 发送 Google `thinking_config.thinking_level=minimal`
  - `gemini-2.5-flash` 发送 Google `thinking_config.thinking_budget=0`

匹配时会去掉 `openai/` 这类 vendor 前缀，且大小写不敏感。官方不允许关闭思考的模型（`kimi-k3`、`kimi-k2.7-code`、MiniMax M2.x）不会附加关闭参数。新增模型请改 `src/subtitle_translator/translation_core/thinking.py`。

部分供应商的强制推理模型不接受关闭参数，此时 API 会拒绝请求；请改用支持非推理模式的模型。

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
EXTERNAL_GLOSSARY_MAX_TERMS=40
```

## CLI

完整参数请看 `translate --help`。

## Codex Skill

这个仓库也内置了一个 Codex skill，目录在 [`skills/subtitle-translator/`](./skills/subtitle-translator/)。

安装到本地 Codex skills 目录：

```bash
mkdir -p ~/.codex/skills
cp -R skills/subtitle-translator ~/.codex/skills/subtitle-translator
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
