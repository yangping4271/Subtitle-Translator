# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Subtitle Translator is a command-line tool that integrates English video transcription and subtitle translation. It provides two main CLI commands:
- `translate`: Full workflow from audio/video to bilingual subtitles
- `transcribe`: Transcription-only workflow

The project is structured as a Python package using `uv` for dependency management and distribution via `uv tool install`.

## Architecture

The codebase has a modular architecture with two core processing engines:

### Main Package Structure
```
src/subtitle_translator/
├── cli.py                    # Main translate command entry point
├── processor.py              # Core file processing logic
├── config_manager.py         # Configuration and API key management
├── env_setup.py             # Environment validation
├── logger.py                # Centralized logging
├── service.py               # High-level service coordination
├── transcription_core/      # Parakeet MLX transcription engine
│   ├── cli.py              # Transcribe command entry point
│   ├── audio.py            # Audio processing and chunking
│   ├── alignment.py        # Word-level timestamp alignment
│   └── ...                 # Other transcription modules
└── translation_core/       # LLM-based translation engine
    ├── config.py          # Language configuration
    ├── spliter.py         # Smart sentence splitting
    ├── aligner.py         # Translation alignment
    └── utils/
        ├── ass_converter.py  # ASS subtitle generation
        └── test_openai.py   # API connectivity testing
```

### Key Components

**Translation Pipeline**: The main workflow processes files through transcription → smart splitting → translation → ASS generation. The `processor.py` orchestrates this flow, handling both audio/video files and existing SRT files.

**Transcription Engine**: Uses Parakeet MLX models optimized for Apple Silicon. Handles automatic audio chunking for long files and supports word-level timestamps.

**Translation Engine**: LLM-based translation supporting multiple models (OpenAI, etc.) with reflection mode for improved quality. Includes smart sentence splitting to optimize translation context.

**Configuration System**: Environment-based configuration with interactive setup via `translate init`. Supports per-model configuration for splitting, translation, and summarization tasks.

## Development Commands

### Installation and Setup
```bash
# Install as a tool
uv tool install .

# Update PATH (required after first install)
uv tool update-shell
source ~/.zshenv  # or restart shell

# Development install with dependencies
uv sync --dev
```

### Running the Application
```bash
# Using installed tool
translate init           # Configure API keys
translate               # Batch process current directory
translate -i video.mp4  # Process single file
transcribe audio.mp3    # Transcription only

# Development mode
uv run python -m subtitle_translator.cli --help
uv run python -m subtitle_translator.transcription_core.cli --help
```

### Testing
```bash
# API connectivity test
uv run python -m subtitle_translator.translation_core.utils.test_openai
```

## Configuration

The application requires API configuration via `.env` file or interactive setup:

### Required Environment Variables
```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here

# Model Configuration
SPLIT_MODEL=gpt-4o-mini      # Sentence splitting
TRANSLATION_MODEL=gpt-4o     # Main translation
SUMMARY_MODEL=gpt-4o-mini    # Content summarization
LLM_MODEL=gpt-4o-mini        # Default fallback
```

### Supported Languages
Target languages use standard codes: `zh` (Chinese), `ja` (Japanese), `ko` (Korean), `en` (English), `fr` (French), etc.

## File Processing Logic

The application processes files with intelligent file discovery:
1. Scans for `.srt`, `.mp3`, `.mp4` files in current directory
2. Groups files by base name (removes language suffixes)
3. Skips files where `.ass` output already exists
4. Prioritizes input: `.srt` > `.mp3` > `.mp4`
5. Generates bilingual `.ass` files and preserves original `.srt`

## Error Handling

The codebase includes comprehensive error handling with specific exception types:
- `SmartSplitError`: Sentence splitting failures
- `TranslationError`: Translation API failures  
- `SummaryError`: Summarization failures

Errors are logged and displayed with user-friendly messages while maintaining detailed logs for debugging.