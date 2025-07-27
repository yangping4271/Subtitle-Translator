# Subtitle Translator

[English](./README.md) | [中文](./shared/docs/README_zh.md)

A comprehensive subtitle translation solution with both **command-line interface** and **native macOS application**. It transcribes English audio/video into subtitles, translates them into multiple languages, and generates bilingual ASS subtitle files.

> ⚠️ **Important**: The transcription function only supports English audio/video. If your video is in another language, please prepare an English SRT subtitle file first.

## 🚀 Quick Start

### Choose Your Interface

**🖥️ macOS Application (Recommended)**
- Native macOS experience with drag & drop
- Real-time progress tracking
- Batch processing with queue management
- System integration (Finder, notifications)

**⌨️ Command Line Interface**
- Powerful automation and scripting
- Batch processing capabilities
- Integration with other tools
- Server/headless environments

## 📦 Installation

### macOS Application
```bash
# Clone the repository
git clone <your-repo-url>
cd Subtitle-Translator

# Build the macOS app
./shared/scripts/build-macos-app.sh

# Launch the app
open build/SubtitleTranslatorApp.app
```

### Command Line Interface
```bash
# Install CLI tools  
cd cli/
uv tool install .

# Update PATH to use the installed tools
uv tool update-shell
# Then restart your shell or run: source ~/.zshenv
```

## 🛠️ Development

### Setup Development Environment
```bash
# Setup both CLI and macOS app environments
./shared/scripts/dev.sh setup
```

### CLI Development
```bash
# Start CLI development mode
./shared/scripts/dev.sh cli-dev

# Test CLI functionality
./shared/scripts/dev.sh test-cli

# Initialize configuration
./shared/scripts/dev.sh cli-dev init
```

### macOS App Development
```bash
# Build and run the Swift app
./shared/scripts/dev.sh app-dev build
./shared/scripts/dev.sh app-dev run

# Open in Xcode
./shared/scripts/dev.sh app-dev xcode
```

## 📁 Project Structure

```
Subtitle-Translator/
├── cli/                          # Python CLI Application
│   ├── src/subtitle_translator/  # Core Python modules
│   │   ├── cli.py               # Main CLI entry point
│   │   ├── transcription_core/  # Speech recognition engine
│   │   └── translation_core/    # LLM translation engine
│   ├── pyproject.toml           # Python project configuration
│   └── uv.lock                  # Python dependencies
├── macos-app/                   # SwiftUI macOS Application
│   ├── SubtitleTranslatorApp/   # Swift source code
│   │   ├── Views/              # SwiftUI views
│   │   ├── Models/             # Data models
│   │   └── Services/           # Business logic
│   ├── Package.swift           # Swift package configuration
│   └── App.swift               # macOS app entry point
├── shared/                     # Shared resources
│   ├── docs/                   # Documentation
│   ├── scripts/                # Build and development scripts
│   └── logs/                   # Application logs
└── README.md                   # This file
```

## ✨ Features

### Core Capabilities
- **🎙️ English Audio Transcription**: Powered by Parakeet MLX model, optimized for Apple Silicon
- **🌍 Multi-language Translation**: Support for 10+ languages using LLM models
- **📺 Bilingual Subtitles**: Automatic ASS subtitle generation with custom styling
- **⚡ Batch Processing**: Process multiple files efficiently
- **🧠 Smart Segmentation**: AI-powered sentence splitting for optimal translation

### macOS Application Features
- **🖱️ Drag & Drop Interface**: Simply drag files to start processing
- **📊 Real-time Progress**: Live progress tracking with detailed status
- **🔄 Queue Management**: Batch processing with intelligent task scheduling  
- **⚙️ Visual Configuration**: Easy setup through native UI
- **🔔 System Integration**: Native notifications and Finder integration

### CLI Features
- **🤖 Automation Ready**: Perfect for scripts and batch operations
- **🔧 Advanced Configuration**: Fine-grained control over all parameters
- **📈 Detailed Logging**: Comprehensive logging for debugging
- **🔄 Resume Support**: Continue interrupted processes

## 🎯 Workflows

### Full Workflow (Audio/Video → Bilingual Subtitles)
```
Audio/Video → Transcribe → English SRT → Translate → Bilingual ASS Subtitles
```

### Translation-Only Workflow (Existing English Subtitles)
```
English SRT → Translate → Bilingual ASS Subtitles
```

### Transcription-Only Workflow
```
Audio/Video → Transcribe → Multiple Output Formats
```

## 🎬 Usage Examples

### macOS Application
1. Launch `SubtitleTranslatorApp.app`
2. Configure your API keys in Settings
3. Drag & drop your media files
4. Select target language and options
5. Click "Process" and monitor progress
6. Find your bilingual subtitles in the output folder

### Command Line Interface

**Configuration**
```bash
cd cli/
translate init  # One-click API key configuration
```

**Basic Usage**
```bash
cd cli/
# Batch process all files in current directory
translate

# Process a single file  
translate -i video.mp4

# Translate to different languages
translate -i video.mp4 -t ja    # Japanese
translate -i video.mp4 -t ko    # Korean  
translate -i video.mp4 -t fr    # French

# Enable reflection mode for higher quality
translate -i video.mp4 -r
```

**Transcription Only**
```bash
cd cli/
# Transcribe single file
transcribe video.mp4

# Transcribe with word-level timestamps
transcribe video.mp4 --timestamps

# Multiple output formats
transcribe video.mp4 --output-format all
```

**Advanced Features**
```bash
cd cli/
# Model management
translate model list                    # List cached models
translate model download                # Pre-download models
translate model info                    # Check model status

# Custom configuration
translate -i video.mp4 -t zh -m gpt-4o -r -d  # Debug mode
```

## 🌐 Supported Formats

### Input Formats
- **Audio**: MP3, WAV, FLAC, M4A, AAC
- **Video**: MP4, MOV, MKV, AVI, WebM  
- **Subtitles**: SRT format

### Output Formats
- **translate**: Generates `.srt` (English) and `.ass` (bilingual) files
- **transcribe**: Supports TXT, SRT, VTT, JSON, and more

### Supported Languages
- **Target Languages**: Chinese (zh), Japanese (ja), Korean (ko), English (en), French (fr), German (de), Spanish (es), Portuguese (pt), Russian (ru), Italian (it), Arabic (ar), Thai (th), Vietnamese (vi)

## ⚙️ Configuration

### Environment Variables
```bash
# OpenAI Configuration
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here

# Model Configuration
SPLIT_MODEL=gpt-4o-mini      # Sentence splitting
TRANSLATION_MODEL=gpt-4o     # Main translation  
SUMMARY_MODEL=gpt-4o-mini    # Content summarization

# Hugging Face Optimization (optional)
HF_ENDPOINT=https://hf-mirror.com  # For users in China
```

### Interactive Setup
Both the macOS app and CLI provide interactive configuration:
- API key management
- Model selection
- Language preferences
- Output directory settings

## 🔧 Technical Architecture

### Dual-Engine Design
- **Transcription Core**: Parakeet MLX for high-performance speech recognition
- **Translation Core**: LLM-powered translation with reflection mode
- **Bridge Layer**: Seamless integration between CLI and GUI

### Key Technologies
- **Apple MLX**: Native Apple Silicon optimization
- **Swift + SwiftUI**: Modern macOS app development
- **Python + uv**: Fast dependency management
- **Multi-threaded Processing**: Efficient batch operations

## 🤝 Contributing

We welcome contributions! Please see our development setup above.

### Development Workflow
1. Fork the repository
2. Set up development environment: `./shared/scripts/dev.sh setup`
3. Make your changes
4. Test both CLI and macOS app: `./shared/scripts/dev.sh test-cli`
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- **Parakeet MLX**: Advanced speech recognition model
- **Apple MLX**: High-performance ML framework
- **OpenAI**: Language model APIs
- **uv**: Modern Python package management