---
name: subtitle-translator
description: 当用户想翻译英文 `.srt` 字幕、批量翻译一个目录、生成双语 `.ass`、初始化或排查 `Subtitle-Translator`/`translate` 配置、补充 `context.txt`/`ctx.txt`、或维护 `terminology.txt` 里的术语与 ASR aliases 时，使用此 skill。无论当前是否在项目仓库内，都应优先用此 skill：它会先发现 `translate` 是否已安装，必要时再定位本地仓库或安装官方仓库。
user_invocable: true
arguments: 字幕文件或目录、目标语言、术语/上下文修正要求，例如 "把 ~/Downloads 里的字幕翻译成中文"、"这个 land chain 应该是 LangChain，修好术语表后重跑"。
---

# Subtitle Translator Skill

这个 skill 是一个可全局安装、可在任意目录运行的操作规程，不依赖“当前就在项目仓库里”。

官方仓库：

```text
https://github.com/yangping4271/Subtitle-Translator
```

## 核心原则

- 优先使用已安装的 `translate` CLI
- 没安装时，再定位本地 checkout
- 本地也没有时，再从官方仓库安装
- 处理真实字幕问题时，术语修正必须落到 `terminology.txt`

## 输入和输出

- 输入只支持英文 `.srt`
- 常规输出：
  - `<basename>.<target>.srt`
  - `<basename>.ass`
- 仅在使用 `--preserve-intermediate` 时保留：
  - `<basename>.en.srt`

## 固定位置

配置文件：

```text
~/.config/subtitle-translator/.env
```

全局术语表：

```text
~/.config/subtitle-translator/terminology.txt
```

局部术语表：

```text
<字幕目录>/terminology.txt
```

局部上下文文件：

```text
<字幕目录>/context.txt
<字幕目录>/ctx.txt
```

## 先解析任务

先判断用户是在要：

- 直接翻译字幕
- 修 `terminology.txt` / `aliases` 后重跑
- 补 `context.txt` / `ctx.txt` 后重跑
- 初始化或排查安装与配置

如果用户只是在问项目怎么用，先回答，不修改文件。

## 先用 skill 自带脚本

这个 skill 不是只有说明文档。优先直接调用 `scripts/` 里的脚本，而不是手写重复命令。

脚本入口：

- `scripts/discover_subtitle_translator.sh`
- `scripts/install_subtitle_translator.sh`
- `scripts/run_translate.sh`

如果需要更细的说明，再读：

- `references/install-and-discovery.md`
- `references/terminology-and-context.md`

## 第一步：发现可用的 Subtitle-Translator

不要默认“当前目录就是仓库”。

优先执行：

```bash
bash scripts/discover_subtitle_translator.sh
```

这个脚本会按顺序检查：

- 已安装的 `translate` CLI
- 当前目录及常见本地 checkout 路径
- 是否存在可用仓库副本

输出会明确告诉你：

- `mode=cli`
- `mode=repo`
- `mode=missing`

如果是 `mode=repo`，会同时给出仓库路径。

### 安装或重装

当发现结果是 `mode=missing`，或者用户明确要求安装/重装时，执行：

```bash
bash scripts/install_subtitle_translator.sh
```

重装：

```bash
bash scripts/install_subtitle_translator.sh --force
```

安装脚本负责：

- 克隆或复用官方仓库
- 执行 `uv tool install -e .` 或 `uv tool install -e . --force`
- 尝试执行最小验证

## 第二步：检查配置

先检查：

```text
~/.config/subtitle-translator/.env
```

若没有，执行：

```bash
translate init
```

关键变量：

```text
OPENAI_BASE_URL=
OPENAI_API_KEY=
SPLIT_MODEL=
TRANSLATION_MODEL=
LLM_MODEL=
```

可选变量：

```text
THREAD_NUM=
TARGET_LANGUAGE=
EXTERNAL_GLOSSARY_ENABLED=true
EXTERNAL_GLOSSARY_DOMAINS=programming,tech,education
EXTERNAL_GLOSSARY_MAX_TERMS=40
```

## 第三步：处理术语和 ASR alias

当用户说“这个词翻错了”“这个词被识别成别的词”“以后都要这样翻”时：

- 先确认标准术语
- 再把真实误识别写入 `aliases`
- 用户主要用简体中文时，优先维护 `[简体中文]`
- 默认优先编辑全局术语表
- 只有该规则只适用于某一批字幕时，才写局部 `terminology.txt`

格式示例：

```text
[简体中文]
LangChain = LangChain | aliases: land chain, lang chain, length chain
DeepLearning.AI = DeepLearning.AI | aliases: deep learn ai, deep learning a i
```

约束：

- 标准术语翻译写在等号右边
- 外部 glossary 不能替代 `aliases`
- 不要只在回答里口头纠正，必须落文件

## 第四步：处理上下文

当字幕主题有课程、产品、公司、专有概念等背景，而字幕文本本身不够判断时：

- 在字幕目录创建或更新 `context.txt`
- 内容要短、准、只写翻译判断需要的信息

## 第五步：预览和执行翻译

默认优先调用脚本：

```bash
bash scripts/run_translate.sh -- -i /path/to/subtitle.srt -t zh
```

脚本会自动：

- 优先使用已安装的 `translate`
- 否则回退到本地仓库 `uv run python -m subtitle_translator.cli`
- 若两者都不可用，则报错提示先安装

### 单文件

```bash
bash scripts/run_translate.sh -- -i /path/to/subtitle.srt -t zh
```

### 批量目录

先 dry-run：

```bash
bash scripts/run_translate.sh -- --input-dir /path/to/subtitles -t zh --dry-run
```

再正式执行：

```bash
bash scripts/run_translate.sh -- --input-dir /path/to/subtitles -t zh
```

### 保留中间文件

```bash
bash scripts/run_translate.sh -- -i /path/to/subtitle.srt -t zh --preserve-intermediate
```

### 分别指定断句模型和翻译模型

```bash
bash scripts/run_translate.sh -- -i /path/to/subtitle.srt -t zh --split-model gpt-4o-mini --translation-model gpt-4o
```

## 第六步：验证

最少检查：

- 命令是否成功退出
- `<basename>.<target>.srt` 是否生成
- `<basename>.ass` 是否生成
- 若开启 `--preserve-intermediate`，`<basename>.en.srt` 是否存在

若是修 alias 后重跑，额外检查：

- 关键术语是否已经改正
- alias 是否已经写入正确语言分区

## 第七步：结果反馈

简洁汇报：

- 使用的是哪一种执行路径：
  - 已安装 `translate`
  - 本地仓库 `uv run`
  - 刚从官方仓库安装
- 处理了哪些字幕
- 输出到了哪里
- 更新了哪些 `terminology.txt` / `aliases` / `context.txt`
- 还有什么未完成项，例如缺 API 配置、模型不可用、输入不是 `.srt`

## 不要这样做

- 不要假设 skill 安装位置就是项目仓库
- 不要把“使用当前仓库”写成前提
- 不要把生成字幕、缓存、日志、外部 glossary 缓存提交到仓库
- 不要把一次性的口头纠正留在对话里而不落 `aliases`
