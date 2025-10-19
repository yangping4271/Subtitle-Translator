# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Subtitle Translator is a command-line tool for English video transcription and multilingual subtitle translation. It provides two main CLI commands:
- `translate`: Full workflow from English audio/video to bilingual subtitles in various target languages
- `transcribe`: English transcription-only workflow

The project is structured as a Python package using `uv` for dependency management and distribution via `uv tool install`. Current version: **0.5.0** (Major upgrade with intelligent NLP-powered sentence segmentation).

**Language Support:**
- **Source Language**: English only (for transcription)
- **Target Languages**: Chinese, Japanese, Korean, French, German, Spanish, Portuguese, Italian, Russian, Arabic, Thai, Vietnamese, and more

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
│   ├── vad_chunker.py      # VAD-based intelligent audio chunking
│   └── ...                 # Other transcription modules
└── translation_core/       # LLM-based translation engine
    ├── config.py          # Language configuration
    ├── spliter.py         # Smart sentence splitting
    ├── split_by_llm.py    # LLM-based sentence segmentation with fallback strategies
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

**Transcription Engine**: Uses Parakeet MLX models optimized for Apple Silicon. Features model caching system (`model_cache.py`) for improved performance, automatic audio chunking for long files, supports word-level timestamps, and includes VAD-based intelligent chunking (`vad_chunker.py`) for optimal speech segmentation.

**Translation Engine**: LLM-based translation supporting multiple models (OpenAI, etc.). Includes intelligent sentence splitting with three-tier fallback strategy: sentence-end punctuation → rule-based matching → forced segmentation. Supports translation optimization features and context-aware processing.

**Universal Subtitle Processing System (v0.4.0)**: Revolutionary upgrade that supports both word-level and segment-level subtitles through a unified processing framework. Key innovations:
- **Intelligent Detection**: Automatically identifies subtitle type (word-level vs segment-level)
- **Phoneme-based Conversion**: Converts segment-level subtitles to word-level using advanced phoneme theory (4 characters = 1 phoneme)
- **Multi-language Support**: Supports Latin scripts, CJK characters, Arabic, Cyrillic, and more
- **Zero API Cost Increase**: Reuses existing batch processing framework after conversion
- **Precise Timestamp Allocation**: More accurate than simple proportional distribution

**Configuration System**: Environment-based configuration with interactive setup via `translate init`. Supports per-model configuration for splitting, translation, and summarization tasks.

### Translation Pipeline Architecture

The project implements a sophisticated **three-stage processing architecture** that ensures high-quality subtitle translation from speech to bilingual output.

#### Architecture Overview

The translation pipeline consists of three critical stages that work together to transform raw transcriptions into professional bilingual subtitles:

1. **Segmentation Stage** - Intelligent text splitting and punctuation restoration
2. **Summarization Stage** - Global analysis and error correction
3. **Translation Stage** - Batch processing with context-aware optimization

This design separates concerns effectively: segmentation handles display constraints, summarization provides global context, and translation focuses on quality output.

#### Segmentation Stage (断句阶段)

**Why Segmentation is Essential:**
- ASR outputs typically lack proper punctuation marks
- Long sentences exceed subtitle display time constraints (viewers need time to read)
- Chinese translations require more screen space than English source
- Technical terms must remain intact for accuracy

**Core Functions:**
- **Punctuation Restoration**: Adds missing periods, commas, question marks based on speech patterns
- **Smart Splitting**: Breaks text at semantic boundaries (max 20 words per segment)
- **Term Protection**: Preserves technical terms, product names, and proper nouns intact
- **Length Balancing**: Optimizes segment lengths for comfortable reading speed
- **Translation Preparation**: Considers target language expansion (Chinese typically 20-30% longer)

**Technical Implementation:**
- Uses LLM model (configured via `SPLIT_MODEL`) for intelligent decision-making
- Processes text with `<br>` delimiters for segment boundaries
- Maintains semantic coherence while respecting display constraints
- Module: `translation_core/spliter.py` with prompt in `SPLIT_SYSTEM_PROMPT`

#### Summarization Stage (总结阶段)

**Why Global Analysis is Necessary:**
- ASR errors are systematic - the same misrecognition repeats throughout
- Proper context requires full document understanding
- Terminology consistency needs global perspective
- Cultural and domain context improves translation quality

**Five Core Functions:**

1. **ASR Error Detection**
   - Identifies phonetic misrecognitions (e.g., "Windsurf" → "WinSurf")
   - Recognizes systematic patterns across entire transcript
   - Validates corrections through multiple occurrences
   - Creates error-correction mapping for consistency

2. **Content Understanding**
   - Identifies video type (tutorial, presentation, interview, etc.)
   - Extracts main topics and key arguments
   - Recognizes technical domain and expertise level
   - Notes cultural references and context-dependent expressions

3. **Terminology Unification**
   - Establishes canonical forms for proper nouns
   - Creates consistent technical term glossary
   - Resolves naming inconsistencies (filename vs. content)
   - Builds "do not translate" term list

4. **Context Construction**
   - Provides domain background information
   - Identifies potential cultural adaptation needs
   - Flags idiomatic expressions requiring special handling
   - Notes speaker style and tone for translation consistency

5. **Translation Guidance**
   - Marks segments needing special attention
   - Provides constraints without specific translations
   - Identifies ambiguous references needing clarification
   - Suggests appropriate formality level

**Technical Implementation:**
- Processes entire subtitle file for global perspective
- Uses LLM model (configured via `SUMMARY_MODEL`)
- Outputs structured JSON with corrections and context
- Module: `translation_core/summarizer.py`

#### Translation Stage (翻译阶段)

**Why Batch Processing:**
- LLM context windows have token limits
- Parallel processing dramatically improves speed
- Failed batches can be retried independently
- Maintains consistency through shared context

**Working Mechanism:**
- **Batch Size**: ~50 subtitle segments per batch
- **Context Injection**: Each batch receives summarization results
- **Error Correction**: Applies global ASR fixes consistently
- **Parallel Execution**: Multiple batches processed simultaneously
- **Fallback Strategy**: Failed batches retry with single-segment processing

**Information Flow:**
```
Summary JSON → Parse corrections & context
             ↓
         Batch 1 → Apply corrections → Translate → Optimize
         Batch 2 → Apply corrections → Translate → Optimize  
         Batch N → Apply corrections → Translate → Optimize
             ↓
         Merge results → Alignment → Output
```

**Technical Implementation:**
- Uses threading for parallel batch processing
- Configured via `TRANSLATION_MODEL` and `thread_num`
- Implements retry logic with fallback strategies
- Module: `translation_core/optimizer.py`

#### Information Flow Design

The pipeline implements a carefully orchestrated data flow:

```
Audio/Video Input
       ↓
   Transcription (Parakeet MLX)
       ↓
   Segmentation (Smart Splitting)
       ↓
   Summarization (Global Analysis) ←──────┐
       ↓                                  │
   [Summary Context]                      │
       ↓                                  │
   ┌─────────────┐                       │
   │  Batch 1    │ → Translation → Quality Check
   │  Batch 2    │ → Translation → Quality Check  
   │  ...        │ → Translation → Quality Check
   │  Batch N    │ → Translation → Quality Check
   └─────────────┘
       ↓
   Result Merging
       ↓
   Timeline Alignment
       ↓
   ASS Generation (Bilingual Output)
```

**Key Design Principles:**
- **Unidirectional flow** prevents circular dependencies
- **Context preservation** ensures consistency across batches
- **Graceful degradation** handles failures without stopping
- **Progressive enhancement** allows quality improvements at each stage

#### Supporting Systems

**Alignment System (`aligner.py`)**
- Matches translated segments with original timestamps
- Handles text reflow from splitting/merging
- Ensures synchronization with video/audio

**ASS Generator (`ass_converter.py`)**
- Creates professional Advanced SubStation Alpha format
- Implements bilingual display (original + translation)
- Configures fonts, colors, and positioning
- Optimizes for readability

**JSON Repair Utility (`json_repair.py`)**
- Fixes malformed LLM JSON outputs
- Handles missing quotes, trailing commas
- Recovers from partial responses
- Ensures robust parsing

**Model Cache System (`model_cache.py`)**
- Implements intelligent two-tier caching architecture for optimal performance
- **Memory Cache**: Single-model in-memory cache with automatic lifecycle management
- **Storage Optimization Cache**: Custom `optimized_models` directory for pre-compiled models
- **Cache Location**: `~/.cache/huggingface/optimized_models/` (custom directory name)
- **Cache Detection**: Enhanced `_find_cached_model` function supports both standard HF cache and storage optimization cache
- **Memory Management**: Automatic release for single-file processing, batch-aware caching for multi-file operations
- **Cache Validation**: Intelligent pre-check system accurately detects cached models and displays file information
- **Performance Benefits**: Reduces model loading time from seconds to milliseconds for cached models

#### Design Advantages

This architecture delivers several critical benefits:

**Consistency & Accuracy**
- Global error correction prevents inconsistent translations
- Unified terminology across entire subtitle file
- Context-aware processing improves accuracy
- Systematic ASR fixes applied uniformly

**Quality & Performance**
- Parallel processing accelerates translation
- Contextual information enhances translation quality
- Intelligent segmentation improves readability

**Reliability & Maintainability**
- Modular design enables independent optimization
- Failed batches don't affect successful ones
- Clear separation of concerns
- Comprehensive error handling

**Scalability & Extensibility**
- Easy to add new language support
- Model upgrades don't require architecture changes
- Processing stages can be enhanced independently
- Supports different quality/speed trade-offs

This sophisticated pipeline architecture ensures professional-quality subtitle translation, transforming raw speech recognition output into polished bilingual subtitles that maintain semantic accuracy, technical precision, and viewing comfort.

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

**⚠️ 唯一可靠的重装方法（UV有缓存问题）：**

```bash
# ✅ 标准重装流程 - 任何代码修改后都用这个
uv tool uninstall subtitle-translator
uv cache clean              # 清理构建缓存（关键！）
uv tool install . --force   # 强制重新编译
```

**为什么必须这样：**
- `uv tool uninstall` 不清理构建缓存
- 不加 `--force` 会复用旧的缓存版本
- 模型缓存（`~/.cache/huggingface/`）不受影响

### Running the Application
```bash
# Using installed tool
translate init           # Configure API keys
translate               # Batch process current directory
translate -i video.mp4  # Process single file
transcribe audio.mp3    # Transcription only

# Common options
translate -i video.mp4 -t zh       # Translate to Simplified Chinese
translate -i video.mp4 -t ja       # Translate to Japanese
translate -i video.mp4 -t ko       # Translate to Korean
translate -i video.mp4 -t fr       # Translate to French
translate -i video.mp4 -d          # Debug mode
transcribe video.mp4 --timestamps  # With word-level timestamps
transcribe video.mp4 --vad          # With VAD intelligent chunking (default)
transcribe video.mp4 --no-vad --chunk-duration 120  # Fixed 2-minute chunks
```

### Development Mode Usage
When developing or debugging, prefer using development mode over installed tools:

```bash
# Use development mode for testing configuration issues
uv run python -m subtitle_translator.cli init

# Use development mode for main commands
uv run python -m subtitle_translator.cli --help
uv run python -m subtitle_translator.transcription_core.cli --help

# Development mode with test file
uv run python -m subtitle_translator.cli -i test.mp4
```

**Advantages of Development Mode:**
- No need to reinstall after code changes
- Better error reporting and stack traces (all logs at INFO level by default)
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

**Source Language:**
- English only (for transcription)

**Target Languages:**
The system supports translation from English to multiple languages:
- **Chinese**: `zh` (Simplified), `zh-cn` (Simplified), `zh-tw` (Traditional)
- **Asian Languages**: `ja` (Japanese), `ko` (Korean), `th` (Thai), `vi` (Vietnamese)
- **European Languages**: `fr` (French), `de` (German), `es` (Spanish), `pt` (Portuguese), `it` (Italian), `ru` (Russian)
- **Other**: `ar` (Arabic), and more

**Note**: The system is designed to transcribe English audio/video and translate to any supported target language.

## File Processing Logic

The application processes files with intelligent file discovery:
1. Scans for `.srt`, `.mp3`, `.mp4` files in current directory
2. Groups files by base name (removes language suffixes)
3. Skips files where `.ass` output already exists
4. Prioritizes input: `.srt` > `.mp3` > `.mp4`
5. Generates bilingual `.ass` files and preserves original `.srt`

## Version Management and Testing

### Version Updates
Current version: **0.5.0** (see `pyproject.toml`)

Recent optimizations include:
- **Simplified Segmentation Strategy**: Three-tier fallback strategy for reliable sentence splitting
- **Punctuation-Based Splitting**: Prioritizes sentence-end punctuation for natural breaks
- **Rule-Based Matching**: Seven-tier priority system for semantic boundary detection
- **Strategy Logging**: Clear log messages showing which segmentation strategy succeeded
- **Code Consolidation**: Removed complex dependencies for better maintainability

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
After making changes, verify functionality using **development mode** (preferred):

```bash
# Verify imports and basic functionality
uv run python -c "from subtitle_translator.config_manager import handle_user_abort, safe_prompt; print('✅ Configuration manager working')"

# Test core components
uv run python -c "from subtitle_translator.processor import process_single_file; print('✅ Core processor available')"

# Validate transcription core
uv run python -c "from subtitle_translator.transcription_core.parakeet import ParakeetModel; print('✅ Transcription engine ready')"
```

#### Efficient Development Workflow

**For Regular Development:**
```bash
# 1. Make code changes
# 2. Test in development mode (no reinstall needed)
uv run python -m subtitle_translator.cli -i test.mp4 -d

# 3. When ready for production testing, do standard reinstall
uv tool uninstall subtitle-translator
uv tool install .
```

**For Dependency Changes:**
```bash
# 1. Update pyproject.toml
# 2. Sync development environment
uv sync --dev

# 3. MUST use force reinstall for tool
uv tool uninstall subtitle-translator
uv cache clean  # Clear UV cache for new dependencies
uv tool install . --force  # ✅ CRITICAL: Always use --force
```

### Critical Lessons: UV Cache Behavior

**Real Case (October 2024):** 代码修改后"标准重装"，但运行时发现用的是2个月前的旧代码。

**根本原因：** `uv tool uninstall` 不清理构建缓存，`uv tool install .` 会复用缓存的旧版本。

**正确做法：**
```bash
# ❌ 错误 - 会用缓存的旧代码
uv tool uninstall subtitle-translator && uv tool install .

# ✅ 正确 - 强制重新编译
uv tool uninstall subtitle-translator && uv cache clean && uv tool install . --force
```

## Major Upgrade v0.4.1: VAD Chunking Fixes

### Critical Bug Fixes

**Problem Addressed:**
The VAD (Voice Activity Detection) system had several critical issues that prevented proper user parameter handling and reliable operation:

1. **Parameter Bypass Issue**: VAD was activating even when users specified explicit chunk durations
2. **Missing Dependencies**: Default ONNX mode failed due to missing `onnxruntime` dependency
3. **Inconsistent Documentation**: Help text and docstrings didn't match actual behavior
4. **Code Quality**: Unused imports and incomplete implementation

**Solution Implementation:**

#### 1. **Fixed VAD Logic** (`parakeet.py:283`)
- **Smart Activation**: VAD now only activates when `chunk_duration < 0` (intelligent mode)
- **User Respect**: Positive chunk_duration values use fixed chunking, ignoring VAD
- **Clear Behavior**: `-1` = VAD smart chunking, `120` = 2-minute fixed chunks, `0` = no chunking

#### 2. **Enhanced Dependencies** (`pyproject.toml`)
- **Added `onnxruntime>=1.22.0`**: Ensures VAD's default ONNX mode works reliably
- **Explicit Declaration**: No more silent fallbacks due to missing dependencies
- **Better Reliability**: First-time VAD usage now works out of the box

#### 3. **Improved CLI Documentation** (`cli.py`)
- **Clear Help Text**: `--chunk-duration` now explains VAD vs fixed chunking behavior
- **User Guidance**: `--vad` option explains when it activates
- **Parameter Interaction**: Users understand the relationship between settings

#### 4. **Code Quality Improvements**
- **Removed Unused Imports**: Cleaned up `numpy` and `Path` imports not being used
- **Corrected Docstrings**: Fixed `use_vad` default value documentation
- **Consistent Behavior**: Code now matches documentation and user expectations

#### Technical Benefits

**Predictable Behavior:**
```bash
# Smart VAD chunking (default)
transcribe video.mp4                    # Uses VAD intelligent segmentation

# Fixed chunking (user specified)
transcribe video.mp4 --chunk-duration 120   # 2-minute fixed chunks, no VAD

# No chunking
transcribe video.mp4 --chunk-duration 0     # Single processing, no chunking
```

**Reliable Dependencies:**
- No more `ModuleNotFoundError: onnxruntime` on first VAD usage
- Consistent performance across different environments
- Proper fallback handling when VAD is unavailable

**Enhanced User Experience:**
- Clear parameter behavior that matches expectations
- Better CLI help text explaining options
- Reliable functionality without hidden dependencies

## Major Upgrade v0.4.0: Universal Subtitle Processing

### Revolutionary Feature: Unified Subtitle Type Processing

**Problem Solved:**
Prior to v0.4.0, the system only handled word-level timestamped subtitles effectively. Segment-level subtitles (typical output from standard transcription) were either skipped or poorly processed, leading to suboptimal viewing experience.

**Solution Implementation:**
Inspired by VideoCaptioner's approach, we implemented a unified processing strategy that converts all subtitle types to a common format before applying intelligent sentence segmentation.

#### Technical Architecture

**Core Strategy: Convert-Then-Process**
```
Input: Any Subtitle Type
  ↓
Detection: is_word_timestamp() 
  ↓
Conversion: split_to_word_segments() [if needed]
  ↓
Unified Processing: merge_segments() [existing framework]
  ↓
Output: Optimized Bilingual Subtitles
```

#### Key Implementation Details

**1. Enhanced Detection System (`data.py:56-76`)**
- **Stricter Criteria**: Changed from ≤4 characters to ≤2 characters per segment
- **Higher Accuracy**: 80% threshold ensures reliable type identification
- **Multi-language Aware**: Handles ASCII and Unicode characters appropriately

**2. Phoneme-based Timestamp Conversion (`data.py:78-164`)**
- **Linguistic Foundation**: 4 characters = 1 phoneme (based on phonetic theory)
- **Multi-language Regex**: Comprehensive pattern matching for:
  - Latin scripts (English, French, German, etc.)
  - CJK characters (Chinese, Japanese, Korean)
  - Cyrillic (Russian, etc.)
  - Arabic, Hebrew, Thai, Hindi, and more
- **Precise Time Allocation**: Proportional distribution based on phoneme count

**3. Unified Processing Pipeline (`service.py:123-157`)**
- **Single Code Path**: All subtitles flow through the same optimized processing
- **Zero API Cost**: Conversion happens locally, reuses existing batch framework
- **Performance Maintained**: Same threading and caching optimizations apply

#### Performance Benefits

**API Call Efficiency:**
- **Before v0.4.0**: Segment-level subtitles → Poor or no processing
- **After v0.4.0**: All subtitle types → Same optimized batch processing (typically 5-8 API calls total)

**Processing Speed:**
- **Conversion Stage**: Local processing, ~1-2 seconds
- **Segmentation Stage**: Leverages existing batch optimization
- **Total Overhead**: <10% increase for significantly better output quality

**Output Quality:**
- **Better Readability**: Long segments split into viewer-friendly sentences
- **Accurate Timing**: Phoneme-based allocation more precise than simple proportion
- **Universal Support**: Works with any transcription source or subtitle format

#### Supported Use Cases

**Now Fully Supported:**
1. **Standard Transcription Output**: `transcribe video.mp4` (segment-level) → Perfect processing
2. **Word-level Transcription**: `transcribe video.mp4 --timestamps` → Enhanced processing (as before)
3. **External SRT Files**: Any `.srt` file → Automatic detection and optimal processing
4. **Mixed Content**: Multi-language subtitles → Intelligent character recognition

**Technical Validation:**
- **Test Case**: 0001_Welcome.mp4 (3-minute video)
- **Input**: 13 long segments (avg 12 seconds each)
- **Output**: 48 optimized sentences (avg 2.5 seconds each)
- **Accuracy**: Perfect timestamp alignment, semantic coherence maintained

### Legacy Improvements (v0.2.x series)

#### Terminal Output Optimization
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
The project now uses an **optimized reinstallation strategy**:

1. **Standard reinstall** for most development (fast and efficient)
2. **Development mode testing** to avoid unnecessary reinstalls  
3. **Deep clean** only when encountering dependency issues
4. **Selective cache cleaning** based on actual needs

### Quality Assurance and Maintenance

#### Pre-Commit Workflow
- **Development mode testing** → **Standard reinstall** → **Production testing** → **Version update** → **Commit code**
- Always prefer development mode (`uv run`) for initial testing
- Use standard reinstall for final validation
- Only use deep clean when encountering specific issues

#### Post-Commit Documentation Sync
**Important**: After each code commit, review and update this CLAUDE.md file to maintain accuracy:
- Check if version number in `pyproject.toml` needs updating in documentation
- Document new features, architectural changes, or module additions  
- Update development commands, configuration requirements, or dependencies
- Record significant performance improvements or bug fixes
- Update reinstallation guidance based on new patterns or issues discovered
- Ensure documentation reflects current codebase state and best practices

This practice ensures that future Claude Code instances always receive the most current and accurate project guidance.