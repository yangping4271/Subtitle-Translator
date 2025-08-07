# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Subtitle Translator is a command-line tool that integrates English video transcription and subtitle translation. It provides two main CLI commands:
- `translate`: Full workflow from audio/video to bilingual subtitles  
- `transcribe`: Transcription-only workflow

The project is structured as a Python package using `uv` for dependency management and distribution via `uv tool install`. Current version: **0.2.5**.

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
│   ├── parakeet.py         # MLX model implementation
│   ├── model_cache.py      # Model caching system
│   └── ...                 # Other transcription modules
└── translation_core/       # LLM-based translation engine
    ├── config.py          # Language configuration
    ├── spliter.py         # Smart sentence splitting
    ├── aligner.py         # Translation alignment
    ├── optimizer.py       # Translation optimization
    ├── summarizer.py      # Content summarization
    └── utils/
        ├── ass_converter.py  # ASS subtitle generation
        ├── json_repair.py   # JSON repair utilities
        └── test_openai.py   # API connectivity testing
```

### Key Components

**Translation Pipeline**: The main workflow processes files through transcription → smart splitting → translation → ASS generation. The `processor.py` orchestrates this flow, handling both audio/video files and existing SRT files.

**Transcription Engine**: Uses Parakeet MLX models optimized for Apple Silicon. Features model caching system (`model_cache.py`) for improved performance, automatic audio chunking for long files, and supports word-level timestamps.

**Translation Engine**: LLM-based translation supporting multiple models (OpenAI, etc.) with reflection mode for improved quality. Includes smart sentence splitting to optimize translation context and translation optimization features.

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

### Reinstallation (Clean Install)
When you need to completely reinstall the project (e.g., after major changes):

```bash
# 1. Uninstall the existing tool
uv tool uninstall subtitle-translator

# 2. Clean all caches and build artifacts
uv cache clean
rm -rf build/ dist/ *.egg-info/ .eggs/ __pycache__/ .pytest_cache/ .coverage src/**/__pycache__/ **/*.pyc

# 3. Reinstall
uv tool install .

# 4. Update PATH if needed (usually not required for reinstalls)
uv tool update-shell
source ~/.zshenv  # or restart shell
```

**Note**: This clean reinstallation process should be performed whenever:
- You've made significant changes to the codebase
- Dependencies have been updated
- You encounter installation-related issues
- You want to ensure a clean environment

### Running the Application
```bash
# Using installed tool
translate init           # Configure API keys
translate               # Batch process current directory
translate -i video.mp4  # Process single file
transcribe audio.mp3    # Transcription only

# Common options
translate -i video.mp4 -t ja -r    # Translate to Japanese with reflection
translate -i video.mp4 -d          # Debug mode
transcribe video.mp4 --timestamps  # With word-level timestamps
```

### Development Mode Usage
When developing or debugging, prefer using development mode over installed tools:

```bash
# Use development mode for testing configuration issues
uv run python -m subtitle_translator.cli init

# Use development mode for main commands
uv run python -m subtitle_translator.cli --help
uv run python -m subtitle_translator.transcription_core.cli --help

# Development mode with custom parameters
uv run python -m subtitle_translator.cli -i test.mp4 -d  # debug mode
```

**Advantages of Development Mode:**
- No need to reinstall after code changes
- Better error reporting and stack traces
- Easier debugging with IDE integration
- Direct access to source code modifications

## Configuration

The application requires API configuration via `.env` file or interactive setup:

### Required Environment Variables
```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here

# Hugging Face Download Configuration (optional)
# Improves model download reliability and speed, especially for users in China
HF_ENDPOINT=https://hf-mirror.com

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

## Version Management and Testing

### Version Updates
Current version: **0.2.5** (see `pyproject.toml`)

Recent optimizations include:
- Enhanced terminal output experience with reduced redundancy
- Model caching improvements
- Better error handling for subtitle processing
- Optimized batch processing workflows

### Testing and Validation
```bash
# Test API connectivity
uv run python -c "from subtitle_translator.translation_core.utils.test_openai import test_openai; test_openai()"

# Verify configuration
uv run python -m subtitle_translator.cli init --test

# Test with existing test files
uv run python -m subtitle_translator.cli -i test_video.mp4 -d
```

The repository includes test files (`test_video.mp4`, `test_video.srt`, `test_video.ass`) for validation.

#### Development Mode Recommendations
For development and testing, **always prefer development mode** over installed tools:

**Why Use Development Mode:**
- ✅ No reinstallation needed after code changes
- ✅ Real-time testing of modifications
- ✅ Better error reporting and debugging
- ✅ Immediate feedback on optimizations
- ✅ Safer testing environment

**Development Commands:**
```bash
# Configuration testing (preferred method)
uv run python -m subtitle_translator.cli init

# Main application testing
uv run python -m subtitle_translator.cli --help
uv run python -m subtitle_translator.cli -i test.mp4 -d

# Transcription testing
uv run python -m subtitle_translator.transcription_core.cli --help
```

#### Code Quality Verification
After making changes, verify functionality:

```bash
# Verify imports and basic functionality
uv run python -c "from subtitle_translator.config_manager import handle_user_abort, safe_prompt; print('✅ Configuration manager working')"

# Test core components
uv run python -c "from subtitle_translator.processor import process_single_file; print('✅ Core processor available')"

# Validate transcription core
uv run python -c "from subtitle_translator.transcription_core.parakeet import ParakeetModel; print('✅ Transcription engine ready')"
```

## Recent Improvements (v0.2.x series)

### Terminal Output Optimization
- Eliminated redundant file name displays during batch processing
- Optimized model loading messages with caching awareness  
- Reduced verbose output for improved user experience
- Better progress reporting for long operations

### Model Caching System
- Intelligent model cache management via `model_cache.py`
- Reduced startup time for repeated transcriptions
- Automatic cache validation and cleanup
- Memory-efficient model loading

### Error Handling and Performance

The system provides enhanced error messages for download failures:
- Network connectivity suggestions
- Mirror configuration recommendations  
- Specific troubleshooting steps for different error types
- Automatic fallback to working mirrors for model downloads

Performance benefits:
- **Reduced Download Failures**: Automatic fallback to working mirrors
- **Improved Speed**: Uses fastest available endpoint based on real-time testing
- **Better User Experience**: Clear progress reporting and helpful error messages
- **Zero Configuration**: Works out of the box with intelligent defaults

## Development Testing and Research

### Testing Workflow
```bash
# Generate test files using ffmpeg (preserve for future use)
ffmpeg -f lavfi -i testsrc2=duration=30:size=1280x720:rate=30 -f lavfi -i sine=frequency=1000:duration=30 -pix_fmt yuv420p test_video.mp4

# Create corresponding SRT subtitle manually for testing
# Test files are preserved in repository: test_video.mp4, test_video.srt, test_video.ass
```

### Reinstallation Process
1. Performs complete uninstall and cleanup
2. Reinstalls with fresh dependencies
3. Tests installation integrity
4. Updates version if needed
5. Commits changes to repository

### Quality Assurance and Maintenance

#### Pre-Commit Workflow
- Test installation → Validate functionality → Update version → Commit code
- Always test with both development mode and installed tool
- Verify all major workflows: transcription-only, translation-only, full pipeline

#### Post-Commit Documentation Sync
**Important**: After each code commit, review and update this CLAUDE.md file to maintain accuracy:
- Check if version number in `pyproject.toml` needs updating in documentation
- Document new features, architectural changes, or module additions
- Update development commands, configuration requirements, or dependencies
- Record significant performance improvements or bug fixes
- Ensure documentation reflects current codebase state

This practice ensures that future Claude Code instances always receive the most current and accurate project guidance.