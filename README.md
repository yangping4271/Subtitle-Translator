# Subtitle Translator

[English](./README.md) | [中文](./README_zh.md)

A command-line tool for translating English `.srt` subtitles into other languages and generating bilingual `.ass` subtitles.

## Highlights

- Translate English subtitles into Chinese, Japanese, Korean, French, and more
- Generate translated `.srt` and bilingual `.ass` output
- Work with OpenAI-compatible APIs, including local services such as LM Studio
- Support extra context via `context.txt` / `ctx.txt`
- Support global and local terminology files for consistent translation

## Install

```bash
git clone https://github.com/yangping4271/Subtitle-Translator.git
cd Subtitle-Translator
uv tool install .
uv tool update-shell
```

Then restart your shell, or run `source ~/.zshenv`.

## Quick Start

Initialize config:

```bash
translate init
```

Translate subtitles:

```bash
# Translate all SRT files in the current directory to Chinese
translate

# Translate one file
translate -i subtitle.srt

# Choose a target language
translate -i subtitle.srt -t ja
translate -i subtitle.srt -t fr

# Keep intermediate files
translate -i subtitle.srt -t zh --preserve-intermediate
```

Source language is English only. Common target language codes include `zh`, `zh-tw`, `ja`, `ko`, `fr`, `de`, `es`, `pt`, `it`, `ru`, `ar`, `th`, and `vi`.

## Configuration

Recommended:

```bash
translate init
```

This creates `~/.config/subtitle-translator/.env`.

Manual configuration also works:

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
SPLIT_MODEL=your-split-model
TRANSLATION_MODEL=your-translation-model
LLM_MODEL=your-default-model
```

For LM Studio or other local OpenAI-compatible services:

```bash
OPENAI_BASE_URL=http://127.0.0.1:1234/v1
OPENAI_API_KEY=
SPLIT_MODEL=your-local-split-model
TRANSLATION_MODEL=your-local-translation-model
LLM_MODEL=your-local-default-model
```

## Optional Context And Terminology

Put `context.txt` or `ctx.txt` next to the subtitle file when you want to provide domain context.

Use terminology files when you need consistent term translation:

- Global: `~/.config/subtitle-translator/terminology.txt`
- Local override: `terminology.txt` in the subtitle file directory

Basic format:

```text
[简体中文]
LLM = 大语言模型 (Large Language Model)
AGI = 通用人工智能 (AGI)
```

Local terminology is merged on top of the global file.

## CLI

Use `translate --help` for the full option list.

## Development

```bash
uv sync --dev
uv run python -m subtitle_translator.cli --help
```

## License

MIT. See [LICENSE](LICENSE).
