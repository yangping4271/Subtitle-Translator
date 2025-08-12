# Subtitle Translator

[English](./README.md) | [中文](./README_zh.md)

A comprehensive subtitle translation system with both **command-line tools** and **Chrome extension** for YouTube. It transcribes English audio/video into subtitles, translates them into multiple languages, and provides bilingual subtitle display.

> ⚠️ **Important**: The transcription function only supports English audio/video. If your video is in another language, please prepare an English SRT subtitle file first.

## 🎯 Two Usage Modes

### 1. Command-Line Tools
Perfect for local file processing with professional-grade quality.

### 2. Chrome Extension for YouTube 🆕
Real-time bilingual subtitles for any YouTube video, with intelligent caching system.

## Features

### Core Subtitle Processing
- **English Video Transcription**: Transcribes English audio/video to SRT subtitles using the Parakeet MLX model.
- **AI-Powered Translation**: Supports various LLM models for translation into multiple languages.
- **Three-Stage Translation**: Smart segmentation → content summarization → batch translation.
- **File Path Context Intelligence**: Leverages file and folder names to improve translation accuracy and terminology consistency.
- **Bilingual Subtitles**: Automatically generates bilingual ASS subtitle files.
- **Batch Processing**: Supports processing multiple files at once.
- **Intermediate File Preservation**: `--preserve-intermediate` option to keep English and translated SRT files.

### Chrome Extension Features 🆕
- **Universal YouTube Support**: Works with any YouTube video, regardless of original subtitle availability.
- **Intelligent Caching**: Three-tier caching system (audio, subtitles, translation) for fast re-processing.
- **Real-time Display**: Seamless bilingual subtitle overlay with zero-delay synchronization.
- **Smart Fallback**: Automatic fallback from cache to real-time processing.
- **Manual Audio Upload**: Support for manual audio upload when auto-download fails.
- **Debug Tools**: Comprehensive logging and status monitoring (Ctrl+D to toggle debug panel).

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

### Command-Line Workflow
#### Full Workflow (translate command)
```
Audio/Video → Transcribe → English SRT → Translate → Bilingual ASS Subtitles
```

#### Translation-Only Workflow (with existing English subtitles)
```
English SRT → Translate → Bilingual ASS Subtitles
```

#### Transcription-Only Workflow (transcribe command)
```
Audio/Video → Transcribe → Multiple Output Formats
```

### Chrome Extension Workflow 🆕
#### First-time Processing
```
YouTube Video → Audio Download → Transcribe → Translate → Cache → Real-time Display
```

#### Cache Hit (Subsequent Access)
```
YouTube Video → Cache Lookup → Instant Load → Real-time Display
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

### Command-Line Configuration
#### Quick Configuration
```bash
translate init
```

The interactive configuration includes:
- API key setup for LLM services
- Model configuration for different tasks
- **Hugging Face mirror configuration** (for improved model download speeds)

### Chrome Extension Setup 🆕
#### 1. Install Chrome Extension
1. Open `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" and select the `chrome-extension` folder (production-ready)
4. Pin the extension to toolbar for easy access

#### 2. Start Backend Service
```bash
# Start the backend server (required for Chrome extension)
uv run python backend/server.py
# Server runs on http://127.0.0.1:9009
```

#### 3. Configure Extension
1. Click the extension icon in Chrome toolbar
2. Enter your OpenAI API key and base URL
3. Save configuration
4. Visit any YouTube video - the extension will automatically start processing

#### 4. Usage Tips
- **First-time processing**: Takes 5-15 minutes (downloads + transcribes + translates)
- **Cached videos**: Load in seconds with high-quality bilingual subtitles
- **Debug panel**: Press `Ctrl+D` to toggle debug information
- **Export logs**: Press `Ctrl+L` to export logs for troubleshooting
- **Manual upload**: If auto-download fails, manually upload audio via API

### Manual Configuration
Create a `.env` file:
```bash
# OpenAI API Configuration (required)
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here

# Hugging Face Mirror Configuration (optional, improves download speed)
# Recommended for users in China or with slow connections to huggingface.co
HF_ENDPOINT=https://hf-mirror.com

# Model Configuration
SPLIT_MODEL=gpt-4o-mini      # Sentence splitting model
TRANSLATION_MODEL=gpt-4o     # Translation model
SUMMARY_MODEL=gpt-4o-mini    # Summarization model
LLM_MODEL=gpt-4o-mini        # Default model
```

### Hugging Face Mirror Configuration

For improved model download reliability and speed, especially for users in China, you can configure a Hugging Face mirror:

#### Option 1: Interactive Configuration
```bash
translate init
# Choose "yes" when prompted about Hugging Face mirror configuration
```

#### Option 2: Environment Variable
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

#### Option 3: Add to .env file
```bash
# Add this line to your .env file
HF_ENDPOINT=https://hf-mirror.com
```

#### Supported Mirrors
- **hf-mirror.com** (Recommended for China users)
- **huggingface.co** (Official, default)
- Custom mirror endpoints

The system automatically detects network connectivity and chooses the best download method:
1. **huggingface-cli** + configured mirror (if available)
2. **hf_hub_download** + configured mirror
3. **Automatic fallback** to alternative mirrors if the primary fails

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