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
LangChain = LangChain | aliases: land chain, lang chain
```

Use `aliases` to hint common speech-recognition mistakes. In the example above,
`land chain` and `lang chain` are treated as likely ASR variants and corrected to
`LangChain` before translation.

Local terminology is merged on top of the global file.

For Simplified Chinese, the tool also enables a local external glossary cache by
default for `programming,tech,education`. External terms are not injected in
full; only terms found in the current subtitle batch are added to the prompt.
You can tune this behavior with:

```bash
EXTERNAL_GLOSSARY_ENABLED=true
EXTERNAL_GLOSSARY_DOMAINS=programming,tech,education
EXTERNAL_GLOSSARY_MAX_TERMS=40
```

## CLI

Use `translate --help` for the full option list.

## Codex Skill

This repo also bundles a Codex skill under [`subtitle-translator/`](./subtitle-translator/).

Install it into your Codex skills directory:

```bash
mkdir -p ~/.codex/skills
cp -R subtitle-translator ~/.codex/skills/subtitle-translator
```

Then restart Codex. After restart, you can invoke it with `$subtitle-translator`.

The skill can:

- discover or install the `translate` CLI when needed
- translate one `.srt` file or a whole directory
- update `terminology.txt` entries and ASR `aliases`
- add or refine `context.txt` / `ctx.txt` before rerunning a translation

## Development

```bash
uv sync --dev
uv run python -m subtitle_translator.cli --help
```

## License

MIT. See [LICENSE](LICENSE).
