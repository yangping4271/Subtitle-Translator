# Subtitle Translator

[English](./README.md) | [中文](./README_zh.md)

A command-line tool for English video transcription and multilingual subtitle translation. It transcribes English audio/video into subtitles, translates them into multiple target languages, and generates bilingual ASS subtitle files.

> ⚠️ **Source Language**: This tool only supports **English** audio/video for transcription. If your content is in another language, please provide English SRT subtitles first.

## Features

- **English Audio/Video Transcription**: Uses Parakeet MLX model to transcribe English speech to SRT subtitles.
- **Multilingual Translation**: Supports translating English subtitles into Chinese, Japanese, Korean, French, and many other languages.
- **Intelligent Segmentation**: Multi-tier fallback strategies including punctuation-based and rule-based semantic splitting.
- **AI-Powered Translation**: Leverages various LLM models for high-quality translation.
- **Bilingual Subtitles**: Automatically generates bilingual ASS subtitle files (English + target language).
- **Batch Processing**: Processes multiple files simultaneously.
- **Modular Configuration**: Configurable models for sentence splitting, translation, and summarization.

## Quick Start

### Installation
```bash
git clone <your-repo-url>
cd Subtitle-Translator
uv tool install .

# Update PATH to use the installed tools
uv tool update-shell
# Then restart your shell or run: source ~/.zshenv
```

### Configuration

Create `.env` file in your project directory:

```bash
# API Base URL
OPENAI_BASE_URL=https://api.openai.com/v1

# API Key
OPENAI_API_KEY=your-api-key-here

# Model Configuration
SPLIT_MODEL=gpt-4o-mini
TRANSLATION_MODEL=gpt-4o
SUMMARY_MODEL=gpt-4o-mini
LLM_MODEL=gpt-4o-mini
```

### Basic Usage
```bash
# Batch process all files in the current directory (translates to Chinese by default)
translate

# Process a single file
translate -i video.mp4

# Translate to different target languages (Chinese, Japanese, Korean, etc.)
translate -i video.mp4 -t zh    # Chinese (Simplified)
translate -i video.mp4 -t ja    # Japanese
translate -i video.mp4 -t ko    # Korean
translate -i video.mp4 -t fr    # French

# Transcribe audio/video only (no translation)
transcribe video.mp4

# Transcribe multiple files
transcribe audio1.mp3 audio2.wav video.mp4

# Generate word-level timestamps
transcribe video.mp4 --timestamps

# Output in multiple formats
transcribe video.mp4 --output-format all
```

## Workflow

### Full Workflow (English transcription + translation)
```
English Audio/Video → Transcribe → English SRT → Translate → Bilingual ASS
```

### Translation-Only (with existing English subtitles)
```
English SRT → Translate to Target Language → Bilingual ASS
```

### Transcription-Only (transcribe command)
```
English Audio/Video → Transcribe → SRT/TXT/VTT/JSON
```

## Supported Formats

### Input Formats
- **Subtitle Files**: `.srt` (skips transcription, directly translates)
- **Audio Formats** (9 formats): `.mp3`, `.m4a`, `.wav`, `.flac`, `.aac`, `.ogg`, `.wma`, `.aiff`, `.opus`
- **Video Formats** (11 formats): `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`, `.m4v`, `.mpeg`, `.mpg`, `.3gp`, `.ts`

**Processing Priority**: `.srt` > audio formats > video formats (audio transcription is faster)

### Output Formats
- **translate**: Generates `.srt` (English) and `.ass` (bilingual) files.
- **transcribe**: Supports various formats like TXT, SRT, VTT, JSON, etc.

## Transcription Features

A professional transcription tool based on the Parakeet MLX model:

- **High Performance**: Excellent performance on Apple Silicon, powered by the Apple MLX framework.
- **Smart Chunking**: Automatically handles long audio files to prevent memory overflow.
- **Precise Timestamps**: Supports word-level timestamps with millisecond accuracy.
- **Batch Processing**: Transcribe multiple audio files at once.

### Advanced Usage
```bash
# Process long audio (automatic chunking)
transcribe long_podcast.mp3 --chunk-duration 120 --overlap-duration 15

# Custom output directory and filename
transcribe interview.mp3 --output-dir ./transcripts --output-template "interview_{filename}"

# High-precision mode
transcribe audio.mp3 --fp32
```

## Command-Line Reference

### translate Command
```bash
translate [OPTIONS] [COMMAND]

Options:
  -i, --input-file FILE    Path to a single file. If not specified, batch processes the current directory.
  -n, --count INTEGER      Maximum number of files to process [default: -1]
  -t, --target_lang TEXT   Target language [default: zh]
  -o, --output_dir PATH    Output directory [default: Current directory]
  --model TEXT             Transcription model
  -m, --llm-model TEXT     LLM model
```

### transcribe Command
```bash
transcribe [OPTIONS] AUDIOS...

Options:
  --model TEXT                    Transcription model [default: mlx-community/parakeet-tdt-0.6b-v3]
  --output-dir PATH               Output directory [default: .]
  --output-format [txt|srt|vtt|json|all]  Output format [default: srt]
  --output-template TEXT          Filename template [default: {filename}]
  --timestamps/--no-timestamps    Output word-level timestamps [default: False]
  --overlap-duration FLOAT        Overlap duration in seconds [default: 15.0]
  -v, --verbose                   Show detailed information
  --fp32/--bf16                   Use FP32 precision [default: bf16]
```

### Supported Languages

**Source Language:**
- English only (for transcription)

**Target Languages:**
Supports translation from English to multiple languages:
- **Chinese**: `zh` (Simplified), `zh-cn` (Simplified), `zh-tw` (Traditional)
- **Asian Languages**: `ja` (Japanese), `ko` (Korean), `th` (Thai), `vi` (Vietnamese)
- **European Languages**: `fr` (French), `de` (German), `es` (Spanish), `pt` (Portuguese), `it` (Italian), `ru` (Russian)
- **Other**: `ar` (Arabic), and more

> **Note**: The system is designed to transcribe English audio/video and translate to any supported target language.

## Configuration

Create `.env` file in the project root directory:
```bash
# OpenAI API Configuration (required)
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here

# Model Configuration
SPLIT_MODEL=gpt-4o-mini      # Sentence splitting model
TRANSLATION_MODEL=gpt-4o     # Translation model
SUMMARY_MODEL=gpt-4o-mini    # Summarization model
LLM_MODEL=gpt-4o-mini        # Default model
```

## Development

```bash
# Install development dependencies
uv sync --dev

# Run the main program
uv run python -m subtitle_translator.cli --help

# Run the transcription feature
uv run python -m subtitle_translator.transcription_core.cli --help
```


## License

MIT License - See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Parakeet MLX](https://github.com/senstella/parakeet-mlx) - An implementation of the Nvidia Parakeet model using MLX on Apple Silicon.
- [Video Captioner](https://github.com/WEIFENG2333/VideoCaptioner) - An intelligent subtitle assistant project.
- [uv](https://github.com/astral-sh/uv) - A modern Python package management tool.
- [Typer](https://github.com/tiangolo/typer) - An excellent command-line interface framework.

---

**📧 Contact**: For questions or suggestions, please contact us via Issues or Pull Requests. 