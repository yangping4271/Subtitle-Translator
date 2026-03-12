# Subtitle Translator

[English](./README.md) | [中文](./README_zh.md)

A command-line tool for multilingual subtitle translation. Translates English SRT subtitles into multiple target languages and generates bilingual ASS subtitle files.

## Features

- **Multilingual Translation**: Supports translating English subtitles into Chinese, Japanese, Korean, French, and many other languages.
- **Intelligent Segmentation**: Multi-tier fallback strategies including punctuation-based and rule-based semantic splitting.
- **AI-Powered Translation**: Leverages various LLM models for high-quality translation.
- **Bilingual Subtitles**: Automatically generates bilingual ASS subtitle files (English + target language).
- **Batch Processing**: Processes multiple files simultaneously.
- **Modular Configuration**: Configurable models for sentence splitting and translation.
- **External Context Support**: Provide additional translation context via context.txt files.
- **Terminology Support**: Supports global and local terminology to ensure consistent translation of technical terms.

## Quick Start

### Installation
```bash
git clone https://github.com/yangping4271/Subtitle-Translator.git
cd Subtitle-Translator
uv tool install .

# Update PATH to use the installed tools
uv tool update-shell
# Then restart your shell or run: source ~/.zshenv
```

### Configuration

**Recommended: Use the interactive setup**

```bash
translate init
```

This will create a configuration file at `~/.config/subtitle-translator/.env` with your API settings.

**Alternative: Set environment variables directly**

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=your-api-key-here
export SPLIT_MODEL=gpt-4o-mini
export TRANSLATION_MODEL=gpt-4o
```

**Alternative: Manually create config file**

Create `~/.config/subtitle-translator/.env`:

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
SPLIT_MODEL=gpt-4o-mini
TRANSLATION_MODEL=gpt-4o
LLM_MODEL=gpt-4o-mini
```

### Basic Usage
```bash
# Batch process all SRT files in the current directory (translates to Chinese by default)
translate

# Process a single file
translate -i subtitle.srt

# Translate to different target languages (Chinese, Japanese, Korean, etc.)
translate -i subtitle.srt -t zh    # Chinese (Simplified)
translate -i subtitle.srt -t ja    # Japanese
translate -i subtitle.srt -t ko    # Korean
translate -i subtitle.srt -t fr    # French

# Preserve intermediate translation files
translate -i subtitle.srt -t zh --preserve-intermediate
```

## Workflow

```
English SRT Subtitle → Intelligent Segmentation → AI Translation → Bilingual ASS Subtitle
```

## Supported Formats

### Input Formats
- **Subtitle Files**: `.srt` (English subtitles)

### Output Formats
- **English SRT**: Original English subtitles
- **Target Language SRT**: Translated subtitles
- **Bilingual ASS**: Bilingual subtitle file (English + target language)

## Command-Line Reference

### translate Command
```bash
translate [OPTIONS]

Options:
  -i, --input-file FILE    Path to a single file. If not specified, batch processes the current directory.
  -n, --count INTEGER      Maximum number of files to process [default: -1]
  -t, --target-lang TEXT   Target language [default: zh]
  -o, --output-dir PATH    Output directory [default: Current directory]
  -m, --llm-model TEXT     LLM model
  --split-model TEXT       Sentence splitting model
  --translation-model TEXT Translation model
  -p, --preserve-intermediate  Preserve intermediate translation files
  --dry-run                Preview mode, show files without processing
```

### Supported Languages

**Source Language:**
- English only

**Target Languages:**
Supports translation from English to multiple languages:
- **Chinese**: `zh` (Simplified), `zh-cn` (Simplified), `zh-tw` (Traditional)
- **Asian Languages**: `ja` (Japanese), `ko` (Korean), `th` (Thai), `vi` (Vietnamese)
- **European Languages**: `fr` (French), `de` (German), `es` (Spanish), `pt` (Portuguese), `it` (Italian), `ru` (Russian)
- **Other**: `ar` (Arabic), and more

> **Note**: This tool focuses on translating English subtitles and requires English SRT subtitle files as input.

## Configuration

### Environment Variables

**Recommended: Use the interactive setup**

```bash
translate init
```

This will create a configuration file at `~/.config/subtitle-translator/.env` with your API settings.

**Alternative: Set environment variables directly**

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=your-api-key-here
export SPLIT_MODEL=gpt-4o-mini
export TRANSLATION_MODEL=gpt-4o
```

**Alternative: Manually create config file**

Create `~/.config/subtitle-translator/.env`:

```bash
# OpenAI API Configuration (required)
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here

# Model Configuration
SPLIT_MODEL=gpt-4o-mini      # Sentence splitting model
TRANSLATION_MODEL=gpt-4o     # Translation model
LLM_MODEL=gpt-4o-mini        # Default model
```

### External Context

Provide additional translation context by creating a `context.txt` or `ctx.txt` file in the same directory as your subtitle file:

```bash
# Create context file
cat > context.txt << 'EOF'
This is a technical tutorial about Google Gemini CLI and AI agents.
Key topics: Gemini API, command-line tools, agent development.
Target audience: Developers and AI practitioners.
EOF

# Translation will automatically use the context
translate -i video.srt -t zh
```

The context information will be passed to the translation model along with filename and path information to improve translation quality and terminology accuracy.

### Terminology Configuration

The project supports custom terminology to ensure consistent translation of technical terms.

**Global Terminology** (user config directory):
```bash
# Create config directory
mkdir -p ~/.config/subtitle-translator

# Edit global terminology
cat > ~/.config/subtitle-translator/terminology.txt << 'EOF'
# Global terminology
[简体中文]
AGI = 通用人工智能 (AGI)
LLM = 大语言模型 (Large Language Model)
Transformer = Transformer

[繁体中文]
AGI = 通用人工智慧 (AGI)
EOF
```

**Local Terminology** (same directory as subtitle file):
```bash
# Create terminology.txt in the subtitle directory
# Local terms will merge with and override global terms
cat > terminology.txt << 'EOF'
[简体中文]
# Override global terms
AGI = 人工通用智能 (AGI)
# Add project-specific terms
project-term = 项目术语
EOF
```

**Format Notes**:
- Supports comment lines starting with `#`
- Use `[Language]` to mark language sections
- Use `Term = Translation` format
- Local terminology merges with global, with local terms taking precedence

## Development

```bash
# Install development dependencies
uv sync --dev

# Run the main program
uv run python -m subtitle_translator.cli --help
```


## License

MIT License - See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Video Captioner](https://github.com/WEIFENG2333/VideoCaptioner) - An intelligent subtitle assistant project.
- [uv](https://github.com/astral-sh/uv) - A modern Python package management tool.
- [Typer](https://github.com/tiangolo/typer) - An excellent command-line interface framework.

---

**📧 Contact**: For questions or suggestions, please contact us via Issues or Pull Requests. 