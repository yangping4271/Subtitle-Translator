# Chrome插件本机启动指南

## 🎯 项目关系说明

### Chrome插件与主项目的关系

这个Chrome插件是 **Subtitle Translator** 项目的浏览器扩展版本，专门为YouTube视频提供实时双语字幕服务。

#### 核心架构关系
```
Subtitle Translator 主项目
├── 核心转录引擎 (transcription_core)     ← Chrome插件调用
│   ├── Parakeet MLX 模型
│   ├── 音频处理和分段
│   └── 词级时间戳对齐
├── 核心翻译引擎 (translation_core)      ← Chrome插件调用  
│   ├── 智能断句 (spliter.py)
│   ├── 内容总结 (summarizer.py)
│   └── 批量优化翻译 (optimizer.py)
├── 命令行工具 (cli.py)                 ← 独立使用
│   ├── translate 命令
│   └── transcribe 命令
└── Chrome插件 (chrome-extension/)      ← 浏览器集成
    ├── 后端API服务 (backend/server.py)
    ├── 前端插件脚本
    └── 实时双语显示
```

#### 功能复用说明

**Chrome插件复用了主项目的核心技术**：

1. **转录技术** ✅
   - 使用相同的 Parakeet MLX 转录引擎
   - 音频分段和处理算法
   - 词级时间戳生成

2. **翻译架构** ✅  
   - 三阶段翻译流程：断句 → 总结 → 批量翻译
   - 智能断句算法（单词级 vs 段落级检测）
   - LLM批量优化翻译

3. **质量保证** ✅
   - 相同的提示词和翻译策略
   - 反思模式和质量优化
   - 术语一致性处理

**Chrome插件的独有功能**：

1. **实时浏览器集成** 🆕
   - YouTube页面自动检测
   - 实时字幕同步显示
   - 浏览器cookies支持

2. **音频缓存系统** 🆕
   - 智能音频文件缓存
   - 手动上传备用方案
   - 缓存管理API

3. **用户交互优化** 🆕
   - 调试面板和日志导出
   - 进度追踪和状态显示
   - 错误处理和用户反馈

### 使用场景对比

| 功能 | 命令行工具 | Chrome插件 |
|------|------------|------------|
| **音频文件转录** | ✅ 本地文件 | ✅ YouTube音频 |
| **视频文件处理** | ✅ 本地视频 | ✅ YouTube视频 |
| **批量文件处理** | ✅ 目录批处理 | ❌ 单视频处理 |
| **实时字幕显示** | ❌ 生成文件 | ✅ 浏览器显示 |
| **缓存和复用** | ❌ 重新处理 | ✅ 智能缓存 |
| **用户交互** | 🔧 命令行 | 🖱️ 图形界面 |

## 📋 准备工作

### 系统要求
- Python 3.10+ 
- Chrome浏览器
- 网络连接（用于API调用和模型下载）
- 磁盘空间：至少2GB（用于模型和音频缓存）

### 必需的API密钥
- OpenAI API密钥（或兼容的API服务）
- 确保API账户有足够余额

## 🔧 本机环境安装

### 步骤1: 使用uv管理项目环境

```bash
# 1. 确保安装了uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或
pip install uv

# 2. 克隆项目（如果还没有）
git clone <项目地址>
cd Subtitle-Translator

# 3. 切换到Chrome插件分支
git checkout youtube-chrome-extension

# 4. 使用uv创建和同步虚拟环境
uv sync --dev

# 5. 安装Chrome插件后端专用依赖
uv add python-multipart

# 6. 验证uv环境
uv run python --version
uv run python -c "import fastapi, uvicorn; print('✅ 依赖安装成功')"

# 7. 安装命令行工具（可选，用于独立使用主项目功能）
uv tool install .
uv tool update-shell
source ~/.zshrc  # 或重启shell

# 测试命令行工具（可选）
translate --help      # 主翻译命令
transcribe --help     # 转录命令
```

### uv环境管理说明

- **虚拟环境**: uv自动创建和管理`.venv`虚拟环境
- **依赖锁定**: `uv.lock`文件确保环境一致性
- **快速安装**: uv比pip快10-100倍
- **版本控制**: 精确的依赖版本管理

### 步骤2: 配置API密钥

#### 方法1: 使用.env文件（推荐）
```bash
# 在项目根目录创建.env文件
cat > .env << 'EOF'
# OpenAI API配置
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# 模型配置
SPLIT_MODEL=gpt-4o-mini
TRANSLATION_MODEL=gpt-4o
SUMMARY_MODEL=gpt-4o-mini
LLM_MODEL=gpt-4o-mini

# Hugging Face下载配置（可选，提高下载速度）
HF_ENDPOINT=https://hf-mirror.com
EOF

# 编辑.env文件，填入你的真实API密钥
nano .env
```

#### 方法2: 使用环境变量
```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# 添加到shell配置文件（永久生效）
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
echo 'export OPENAI_BASE_URL="https://api.openai.com/v1"' >> ~/.zshrc
source ~/.zshrc
```

#### 方法3: 交互式配置
```bash
# 使用项目内置配置工具
uv run python -m subtitle_translator.cli init
```

### 步骤3: 验证uv环境和安装

```bash
# 1. 验证uv环境
uv --version
uv python list  # 查看可用Python版本

# 2. 检查项目依赖
uv tree  # 显示依赖树
uv pip list  # 列出已安装包

# 3. 测试后端服务启动
uv run uvicorn backend.server:app --help

# 4. 测试API连接
uv run python -c "
from subtitle_translator.translation_core.utils.test_openai import test_openai
test_openai()
"

# 5. 验证关键依赖
uv run python -m yt_dlp --version
uv run python -c "import uvicorn; print(f'Uvicorn: {uvicorn.__version__}')"
uv run python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
```

## 🚀 使用uv启动后端服务

### 方法1: uv + uvicorn启动（推荐）

```bash
# 开发模式（自动重载，推荐）
uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009 --reload

# 生产模式（稳定运行）
uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009

# 指定日志级别
uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009 --reload --log-level debug
```

**uv + uvicorn的优势**：
- ✅ **环境隔离**: 自动使用项目虚拟环境
- ✅ **依赖保证**: 确保所有依赖版本正确
- ✅ **热重载**: 代码修改自动生效
- ✅ **快速启动**: uv优化的启动速度
- ✅ **标准方式**: FastAPI官方推荐

### 方法2: uv直接启动

```bash
# 进入后端目录
cd backend

# 使用uv直接运行
uv run python server.py

# 或指定Python版本
uv run --python 3.11 python server.py
```

### uv环境管理命令

```bash
# 查看当前环境状态
uv pip list
uv pip show fastapi uvicorn yt-dlp

# 更新依赖
uv sync
uv add package_name@latest

# 查看虚拟环境路径
uv venv --show-path

# 激活虚拟环境（如需手动操作）
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate  # Windows
```

### 验证服务启动

```bash
# 1. 检查服务健康状态
curl http://127.0.0.1:9009/health

# 应该返回类似以下内容：
# {
#   "ok": true,
#   "python": "3.13.3",
#   "fastapi": "0.116.1",
#   "uvicorn": "0.35.0",
#   "yt_dlp": "2025.08.11"
# }

# 2. 测试下载功能
curl -X POST http://127.0.0.1:9009/test_download

# 3. 查看缓存状态
curl http://127.0.0.1:9009/cache/status
```

## 🎛️ Chrome插件安装

### 步骤1: 加载插件

1. 打开Chrome浏览器
2. 访问 `chrome://extensions/`
3. 开启右上角的"开发者模式"
4. 点击"加载已解压的扩展程序"
5. 选择项目中的 `chrome-extension` 文件夹

### 步骤2: 配置插件

1. 点击Chrome工具栏中的插件图标
2. 填入配置信息：
   ```
   API Base URL: https://api.openai.com/v1
   API Key: your-openai-api-key
   目标语言: 简体中文
   模型: gpt-4o-mini
   ```
3. 点击"保存设置"

### 步骤3: 验证插件

1. 打开任意YouTube视频
2. 检查插件状态显示（右上角调试面板）
3. 按F12查看Console是否有错误
4. 按Ctrl+L可导出详细日志

## 🔧 常见启动问题解决

### 问题1: 端口被占用

```bash
# 查看端口占用
lsof -i :9009

# 杀死占用进程
sudo kill -9 <PID>

# 或使用其他端口
uv run uvicorn backend.server:app --port 9010
```

### 问题2: 依赖缺失

```bash
# 重新安装依赖
uv sync --dev

# 安装额外依赖
uv add python-multipart fastapi uvicorn yt-dlp

# 检查特定依赖
uv run python -c "import fastapi; print(fastapi.__version__)"
```

### 问题3: API密钥问题

```bash
# 验证API密钥
curl -H "Authorization: Bearer your-api-key" \
  https://api.openai.com/v1/models

# 检查环境变量
echo $OPENAI_API_KEY

# 重新配置
uv run python -m subtitle_translator.cli init
```

### 问题4: yt-dlp下载问题

```bash
# 更新yt-dlp
uv add yt-dlp@latest

# 测试下载功能
uv run python -m yt_dlp --cookies-from-browser chrome \
  --skip-download --write-auto-subs --sub-lang en \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 检查Chrome cookies访问
ls -la "$HOME/Library/Application Support/Google/Chrome/Default/Cookies"
```

### 问题5: 权限问题

```bash
# 检查缓存目录权限
ls -la /tmp/yt_cache/

# 创建缓存目录
mkdir -p /tmp/yt_cache/audio /tmp/yt_cache/subtitles
chmod 755 /tmp/yt_cache/

# 修复Python包权限
chmod +x .venv/bin/python
```

## 📱 使用流程

### 完整工作流

1. **启动后端服务**
   ```bash
   # 在项目根目录（推荐方式）
   uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009 --reload
   ```

2. **确认服务运行**
   ```bash
   curl http://127.0.0.1:9009/health
   ```

3. **配置Chrome插件**
   - 加载插件到Chrome
   - 配置API密钥和设置

4. **访问YouTube视频**
   - 打开任意YouTube视频
   - 等待插件处理（5-15分钟首次处理）
   - 观看双语字幕

### 日常使用

```bash
# 1. 启动服务（每次使用前）
uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009 --reload

# 2. 查看处理状态（可选）
curl http://127.0.0.1:9009/cache/status

# 3. 停止服务（使用完毕后）
# Ctrl+C 停止服务
```

## 🛠️ 开发和调试

### 开发模式启动

```bash
# 开发模式（自动重载）
uv run uvicorn backend.server:app --reload --port 9009

# 查看实时日志
tail -f /tmp/backend.log
```

### 调试技巧

```bash
# 1. 导出Chrome插件日志
# 在YouTube页面按Ctrl+L

# 2. 查看后端详细日志
uv run python backend/server.py --debug

# 3. 测试特定功能
curl -X POST http://127.0.0.1:9009/verify_ytdlp \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'

# 4. 手动测试音频缓存
curl -X POST http://127.0.0.1:9009/cache/upload_audio/test \
  -F "file=@test.m4a"
```

### 性能监控

```bash
# 监控系统资源
top -p $(pgrep -f "python.*server.py")

# 监控缓存使用
du -sh /tmp/yt_cache/

# 监控API调用
tail -f ~/.cache/subtitle_translator/logs/
```

## 📋 启动检查清单

使用前请确认以下项目：

- [ ] Python 3.10+已安装
- [ ] 项目依赖已安装（`uv sync --dev`）
- [ ] API密钥已配置
- [ ] 后端服务启动成功（端口9009）
- [ ] Chrome插件已加载并配置
- [ ] 网络连接正常
- [ ] 磁盘空间充足（>2GB）

### 快速验证脚本

```bash
#!/bin/bash
# quick_check.sh - 快速环境检查

echo "🔍 检查Chrome插件环境..."

# 检查Python版本
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2)
echo "Python版本: ${python_version:-未安装}"

# 检查项目依赖
if uv run python -c "import fastapi, uvicorn, yt_dlp" 2>/dev/null; then
    echo "✅ 项目依赖完整"
else
    echo "❌ 项目依赖缺失"
fi

# 检查API配置
if [ -f ".env" ] && grep -q "OPENAI_API_KEY" .env; then
    echo "✅ API配置文件存在"
else
    echo "❌ API配置缺失"
fi

# 检查服务状态
if curl -s http://127.0.0.1:9009/health >/dev/null 2>&1; then
    echo "✅ 后端服务运行中"
else
    echo "❌ 后端服务未启动"
fi

# 检查缓存目录
if [ -d "/tmp/yt_cache" ]; then
    echo "✅ 缓存目录存在"
else
    echo "⚠️ 缓存目录不存在（首次运行时自动创建）"
fi

echo "检查完成！"
```

保存为可执行文件并运行：
```bash
chmod +x quick_check.sh
./quick_check.sh
```

通过这个完整的启动指南，用户应该能够顺利启动和使用Chrome插件了！