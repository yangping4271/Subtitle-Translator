#!/bin/bash

# 开发便利脚本
# 用于快速启动开发环境和测试

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI_DIR="$PROJECT_ROOT/cli"
MACOS_APP_DIR="$PROJECT_ROOT/macos-app"

# 颜色定义
GREEN='\\033[0;32m'
BLUE='\\033[0;34m'
YELLOW='\\033[1;33m'
NC='\\033[0m'

show_help() {
    echo "Subtitle Translator 开发工具"
    echo
    echo "用法: $0 <command> [options]"
    echo
    echo "命令:"
    echo "  cli-dev       启动CLI开发模式"
    echo "  app-dev       启动SwiftUI应用开发模式"
    echo "  test-cli      测试CLI功能"
    echo "  setup         设置开发环境"
    echo "  clean         清理构建文件"
    echo "  help          显示此帮助信息"
    echo
}

setup_dev_env() {
    echo -e "${BLUE}🛠️  设置开发环境...${NC}"
    
    # 设置CLI环境
    echo "设置Python CLI环境..."
    cd "$CLI_DIR"
    if [ ! -f "uv.lock" ]; then
        echo "初始化Python项目..."
        uv sync
    else
        echo "更新Python依赖..."
        uv sync
    fi
    
    # 设置Swift环境
    echo "设置Swift环境..."
    cd "$MACOS_APP_DIR"
    swift package resolve
    
    echo -e "${GREEN}✅ 开发环境设置完成${NC}"
}

cli_dev() {
    echo -e "${BLUE}🐍 启动CLI开发模式...${NC}"
    cd "$CLI_DIR"
    
    case "${1:-}" in
        "init")
            echo "初始化配置..."
            uv run python -m subtitle_translator.cli init
            ;;
        "test")
            echo "测试CLI..."
            uv run python -m subtitle_translator.cli --help
            ;;
        "translate")
            shift
            echo "执行翻译: $@"
            uv run python -m subtitle_translator.cli "$@"
            ;;
        "transcribe")
            shift
            echo "执行转录: $@"
            uv run python -m subtitle_translator.transcription_core.cli "$@"
            ;;
        *)
            echo "CLI开发模式命令:"
            echo "  $0 cli-dev init        - 初始化配置"
            echo "  $0 cli-dev test        - 测试CLI"
            echo "  $0 cli-dev translate   - 执行翻译"
            echo "  $0 cli-dev transcribe  - 执行转录"
            ;;
    esac
}

app_dev() {
    echo -e "${BLUE}🔨 启动SwiftUI应用开发模式...${NC}"
    cd "$MACOS_APP_DIR"
    
    case "${1:-}" in
        "build")
            echo "构建应用..."
            swift build
            ;;
        "run")
            echo "运行应用..."
            swift run
            ;;
        "test")
            echo "运行测试..."
            swift test
            ;;
        "xcode")
            echo "在Xcode中打开..."
            if [ -f "SubtitleTranslatorApp.xcodeproj/project.pbxproj" ]; then
                open SubtitleTranslatorApp.xcodeproj
            else
                echo "生成Xcode项目..."
                swift package generate-xcodeproj
                open SubtitleTranslatorApp.xcodeproj
            fi
            ;;
        *)
            echo "SwiftUI应用开发命令:"
            echo "  $0 app-dev build  - 构建应用"
            echo "  $0 app-dev run    - 运行应用"
            echo "  $0 app-dev test   - 运行测试"
            echo "  $0 app-dev xcode  - 在Xcode中打开"
            ;;
    esac
}

test_cli() {
    echo -e "${BLUE}🧪 测试CLI功能...${NC}"
    cd "$CLI_DIR"
    
    echo "测试基本功能..."
    
    # 测试帮助信息
    echo "1. 测试帮助信息"
    uv run python -m subtitle_translator.cli --help
    
    echo
    echo "2. 测试转录帮助"
    uv run python -m subtitle_translator.transcription_core.cli --help
    
    echo
    echo "3. 测试模型管理"
    uv run python -m subtitle_translator.cli model list
    
    echo -e "${GREEN}✅ CLI测试完成${NC}"
}

clean_all() {
    echo -e "${BLUE}🧹 清理构建文件...${NC}"
    
    # 清理Python缓存
    find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    
    # 清理Swift构建文件
    cd "$MACOS_APP_DIR"
    swift package clean 2>/dev/null || true
    rm -rf .build 2>/dev/null || true
    
    # 清理构建目录
    rm -rf "$PROJECT_ROOT/build" 2>/dev/null || true
    
    echo -e "${GREEN}✅ 清理完成${NC}"
}

# 主逻辑
case "${1:-}" in
    "setup")
        setup_dev_env
        ;;
    "cli-dev")
        shift
        cli_dev "$@"
        ;;
    "app-dev")
        shift
        app_dev "$@"
        ;;
    "test-cli")
        test_cli
        ;;
    "clean")
        clean_all
        ;;
    "help"|"--help"|"-h"|"")
        show_help
        ;;
    *)
        echo "未知命令: $1"
        echo "使用 '$0 help' 查看帮助信息"
        exit 1
        ;;
esac