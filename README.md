# Subtitle Translator

[English](./README.md) | [中文](./README_zh.md)

A command-line tool that integrates English video transcription, subtitle translation. It transcribes English audio/video into subtitles, translates them into multiple languages, and generates bilingual ASS subtitle files.

> ⚠️ **Important**: The transcription function only supports English audio/video. If your video is in another language, please prepare an English SRT subtitle file first.

## Features

- **English Video Transcription**: Transcribes English audio/video to SRT subtitles using the Parakeet MLX model.
- **AI-Powered Translation**: Supports various LLM models for translation into multiple languages.
- **Bilingual Subtitles**: Automatically generates bilingual ASS subtitle files.
- **Batch Processing**: Supports processing multiple files at once.
- **Modular Configuration**: Allows configuring different models for sentence splitting, translation, and summarization.

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
```bash
translate init  # One-click API key configuration
```

### Basic Usage
```bash
# Batch process all files in the current directory (translates to Chinese by default)
translate

# Process a single file
translate -i video.mp4

# Translate to other languages
translate -i video.mp4 -t ja

# Enable reflection mode for higher quality translation
translate -i video.mp4 -r

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

### Full Workflow (translate command)
```
Audio/Video → Transcribe → English SRT → Translate → Bilingual ASS Subtitles
```

### Translation-Only Workflow (with existing English subtitles)
```
English SRT → Translate → Bilingual ASS Subtitles
```

### Transcription-Only Workflow (transcribe command)
```
Audio/Video → Transcribe → Multiple Output Formats
```

## Supported Formats

### Input Formats
- **Audio**: MP3, WAV, FLAC, M4A, AAC, etc.
- **Video**: MP4, MOV, MKV, AVI, WebM, etc.
- **Subtitles**: SRT format

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
  -r, --reflect            Enable reflection translation mode
  -d, --debug              Debug mode
  
Commands:
  init                     Initialize configuration
```

### transcribe Command
```bash
transcribe [OPTIONS] AUDIOS...

Options:
  --model TEXT                    Transcription model [default: mlx-community/parakeet-tdt-0.6b-v2]
  --output-dir PATH               Output directory [default: .]
  --output-format [txt|srt|vtt|json|all]  Output format [default: srt]
  --output-template TEXT          Filename template [default: {filename}]
  --timestamps/--no-timestamps    Output word-level timestamps [default: False]
  --chunk-duration FLOAT          Chunk duration in seconds [default: 120.0]
  --overlap-duration FLOAT        Overlap duration in seconds [default: 15.0]
  -v, --verbose                   Show detailed information
  --fp32/--bf16                   Use FP32 precision [default: bf16]
```

### Supported Translation Languages
Supports translation into multiple languages. Common language codes: `zh` (Chinese), `ja` (Japanese), `ko` (Korean), `en` (English), `fr` (French), etc.

## Configuration

### Quick Configuration
```bash
translate init
```

### Manual Configuration
Create a `.env` file:
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