#!/bin/bash

# macOS应用构建脚本
# 此脚本用于构建包含Python CLI的完整macOS应用

set -e  # 遇到错误时退出

# 颜色定义
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# 项目路径
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLI_DIR="$PROJECT_ROOT/cli"
MACOS_APP_DIR="$PROJECT_ROOT/macos-app"
BUILD_DIR="$PROJECT_ROOT/build"

# 配置
APP_NAME="SubtitleTranslatorApp"
BUNDLE_ID="com.subtitletranslator.app"

echo -e "${BLUE}🚀 开始构建 Subtitle Translator macOS 应用${NC}"
echo "项目根目录: $PROJECT_ROOT"
echo

# 检查必要的工具
check_requirements() {
    echo -e "${BLUE}📋 检查构建环境...${NC}"
    
    # 检查 uv
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ uv 未安装。请先安装 uv: https://docs.astral.sh/uv/getting-started/installation/${NC}"
        exit 1
    fi
    
    # 检查 swift
    if ! command -v swift &> /dev/null; then
        echo -e "${RED}❌ Swift 未安装。请安装 Xcode 命令行工具${NC}"
        exit 1
    fi
    
    # 检查 Python CLI 目录
    if [ ! -d "$CLI_DIR" ]; then
        echo -e "${RED}❌ CLI 目录不存在: $CLI_DIR${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ 构建环境检查通过${NC}"
    echo
}

# 构建 Python CLI
build_cli() {
    echo -e "${BLUE}🐍 构建 Python CLI...${NC}"
    
    cd "$CLI_DIR"
    
    # 确保虚拟环境存在并安装依赖
    echo "安装 Python 依赖..."
    uv sync
    
    # 测试 CLI 是否正常工作
    echo "测试 CLI 功能..."
    if ! uv run python -m subtitle_translator.cli --help > /dev/null; then
        echo -e "${RED}❌ CLI 测试失败${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Python CLI 构建完成${NC}"
    echo
}

# 准备应用资源
prepare_app_resources() {
    echo -e "${BLUE}📦 准备应用资源...${NC}"
    
    # 创建构建目录
    mkdir -p "$BUILD_DIR"
    
    # 创建应用Bundle目录结构
    APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"
    CONTENTS_DIR="$APP_BUNDLE/Contents"
    RESOURCES_DIR="$CONTENTS_DIR/Resources"
    MACOS_DIR="$CONTENTS_DIR/MacOS"
    
    mkdir -p "$RESOURCES_DIR" "$MACOS_DIR"
    
    # 复制Python CLI到Resources
    echo "复制 Python CLI..."
    # 使用rsync以保持符号链接和权限
    rsync -a --delete "$CLI_DIR/" "$RESOURCES_DIR/cli/"
    
    # 确保虚拟环境正确设置
    echo "验证虚拟环境..."
    if [ -d "$RESOURCES_DIR/cli/.venv" ]; then
        # 测试虚拟环境中的Python是否可用
        if "$RESOURCES_DIR/cli/.venv/bin/python" -c "import sys; print('Python OK')" &> /dev/null; then
            echo -e "${GREEN}✅ 虚拟环境验证通过${NC}"
        else
            echo -e "${YELLOW}⚠️ 虚拟环境可能需要重新创建${NC}"
            # 在应用资源目录中重新同步依赖
            cd "$RESOURCES_DIR/cli"
            uv sync
            cd -
        fi
        
        # 测试必要的模块
        echo "验证Python依赖..."
        if "$RESOURCES_DIR/cli/.venv/bin/python" -c "import typer; import mlx; print('Dependencies OK')" &> /dev/null; then
            echo -e "${GREEN}✅ Python依赖验证通过${NC}"
        else
            echo -e "${RED}❌ Python依赖缺失，尝试重新安装...${NC}"
            cd "$RESOURCES_DIR/cli"
            uv sync
            cd -
        fi
    else
        echo -e "${RED}❌ 虚拟环境未找到${NC}"
        exit 1
    fi
    
    # 创建Info.plist
    cat > "$CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>zh_CN</string>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Subtitle Translator</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeExtensions</key>
            <array>
                <string>mp4</string>
                <string>mp3</string>
                <string>wav</string>
                <string>srt</string>
            </array>
            <key>CFBundleTypeName</key>
            <string>Media Files</string>
            <key>CFBundleTypeRole</key>
            <string>Editor</string>
        </dict>
    </array>
</dict>
</plist>
EOF
    
    echo -e "${GREEN}✅ 应用资源准备完成${NC}"
    echo
}

# 构建 Swift 应用
build_swift_app() {
    echo -e "${BLUE}🔨 构建 Swift 应用...${NC}"
    
    cd "$MACOS_APP_DIR"
    
    # 使用swift build构建
    swift build -c release
    
    # 复制可执行文件到应用Bundle
    EXECUTABLE_PATH="$(swift build -c release --show-bin-path)/$APP_NAME"
    if [ -f "$EXECUTABLE_PATH" ]; then
        cp "$EXECUTABLE_PATH" "$BUILD_DIR/$APP_NAME.app/Contents/MacOS/"
        chmod +x "$BUILD_DIR/$APP_NAME.app/Contents/MacOS/$APP_NAME"
    else
        echo -e "${RED}❌ 可执行文件未找到: $EXECUTABLE_PATH${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Swift 应用构建完成${NC}"
    echo
}

# 代码签名（可选）
code_sign() {
    if [ -n "$CODE_SIGN_IDENTITY" ]; then
        echo -e "${BLUE}🔐 代码签名...${NC}"
        codesign --force --deep --sign "$CODE_SIGN_IDENTITY" "$BUILD_DIR/$APP_NAME.app"
        echo -e "${GREEN}✅ 代码签名完成${NC}"
    else
        echo -e "${YELLOW}⚠️  跳过代码签名（未设置CODE_SIGN_IDENTITY环境变量）${NC}"
    fi
    echo
}

# 创建DMG包（可选）
create_dmg() {
    if command -v create-dmg &> /dev/null; then
        echo -e "${BLUE}📦 创建 DMG 安装包...${NC}"
        
        DMG_NAME="SubtitleTranslator-v1.0.0.dmg"
        create-dmg \\
            --volname "Subtitle Translator" \\
            --window-pos 200 120 \\
            --window-size 600 400 \\
            --icon-size 100 \\
            --icon "$APP_NAME.app" 175 120 \\
            --hide-extension "$APP_NAME.app" \\
            --app-drop-link 425 120 \\
            "$BUILD_DIR/$DMG_NAME" \\
            "$BUILD_DIR/$APP_NAME.app"
        
        echo -e "${GREEN}✅ DMG 创建完成: $BUILD_DIR/$DMG_NAME${NC}"
    else
        echo -e "${YELLOW}⚠️  跳过DMG创建（create-dmg未安装）${NC}"
    fi
    echo
}

# 显示构建结果
show_results() {
    echo -e "${GREEN}🎉 构建完成！${NC}"
    echo
    echo "构建产物:"
    echo "  应用Bundle: $BUILD_DIR/$APP_NAME.app"
    if [ -f "$BUILD_DIR/SubtitleTranslator-v1.0.0.dmg" ]; then
        echo "  安装包: $BUILD_DIR/SubtitleTranslator-v1.0.0.dmg"
    fi
    echo
    echo "运行应用:"
    echo "  open '$BUILD_DIR/$APP_NAME.app'"
    echo
}

# 主流程
main() {
    check_requirements
    build_cli
    prepare_app_resources
    build_swift_app
    code_sign
    create_dmg
    show_results
}

# 处理命令行参数
case "${1:-}" in
    "clean")
        echo -e "${BLUE}🧹 清理构建目录...${NC}"
        rm -rf "$BUILD_DIR"
        echo -e "${GREEN}✅ 清理完成${NC}"
        ;;
    "cli-only")
        check_requirements
        build_cli
        ;;
    "app-only")
        build_swift_app
        ;;
    *)
        main
        ;;
esac