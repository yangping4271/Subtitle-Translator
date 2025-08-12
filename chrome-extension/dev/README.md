# YouTube双语字幕翻译器

一个Chrome浏览器插件，**无论YouTube视频是否有原生字幕，都能提供高质量的双语字幕显示**。基于本项目的专业音频转录和翻译技术，为任何YouTube视频生成准确的双语字幕。

## 功能特性

- ✅ **统一转录处理**: 不依赖YouTube原生字幕，始终使用本项目转录技术
- ✅ **高质量翻译**: 集成三阶段翻译架构（断句→总结→批量翻译）
- ✅ **音频缓存系统**: 智能缓存已下载音频，支持手动上传
- ✅ **实时双语显示**: 零延迟字幕同步，流畅观看体验
- ✅ **智能进度管理**: 详细处理状态显示，支持超时保护
- ✅ **多语言支持**: 支持中文、日文、韩文等多种目标语言
- ✅ **调试工具**: 完整的日志系统和状态监控
- ✅ **浏览器Cookie支持**: 使用Chrome cookies绕过地域限制

## 安装方法

### 1. 下载插件文件
下载所有插件文件到本地文件夹：
- `manifest.json`
- `content.js`
- `popup.html`
- `popup.js`
- `style.css`

### 2. 在Chrome中加载插件
1. 打开Chrome浏览器
2. 访问 `chrome://extensions/`
3. 开启右上角的"开发者模式"
4. 点击"加载已解压的扩展程序"
5. 选择包含插件文件的文件夹

### 3. 配置API设置
1. 点击Chrome工具栏中的插件图标
2. 填入你的OpenAI API信息：
   - **API Base URL**: `https://api.openai.com/v1`
   - **API Key**: 你的OpenAI API密钥
   - **目标语言**: 选择翻译目标语言（默认简体中文）
   - **模型**: 选择翻译模型（推荐gpt-4o-mini）
3. 点击"保存设置"

## 使用方法

### 基本使用流程

1. **启动后端服务**
   ```bash
   # 推荐方式（uvicorn）
   uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009 --reload
   
   # 备选方式（Python脚本）
   uv run python backend/server.py
   ```

2. **访问YouTube视频**
   - 打开任何YouTube视频页面
   - 插件会自动开始处理（无需开启CC字幕）
   - 等待后端完成音频下载和转录翻译

3. **观看双语字幕**
   - 双语字幕会自动显示在视频底部
   - 支持实时同步显示

### 手动音频缓存功能

当自动下载失败时，可以手动下载音频并上传到缓存：

#### 方式1: 使用yt-dlp手动下载
```bash
# 下载音频（推荐）
yt-dlp --extract-audio --audio-format m4a --cookies-from-browser chrome \
  --geo-bypass -o "~/Downloads/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=VIDEO_ID"

# 或下载视频（音频不可用时）
yt-dlp --format "worst[height<=480]/worst" --cookies-from-browser chrome \
  --geo-bypass -o "~/Downloads/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=VIDEO_ID"
```

#### 方式2: 使用其他工具下载
- 使用4K Video Downloader、JDownloader等工具
- 确保下载为音频格式（m4a、mp3、wav、aac等）
- 文件名包含YouTube视频ID（可选，便于识别）

#### 上传到缓存
1. **通过API上传**（推荐）
   ```bash
   # 替换VIDEO_ID为实际的YouTube视频ID
   curl -X POST "http://127.0.0.1:9009/cache/upload_audio/VIDEO_ID" \
     -F "file=@/path/to/audio/file.m4a"
   ```

2. **检查缓存状态**
   ```bash
   # 查看指定视频的缓存
   curl "http://127.0.0.1:9009/cache/check/VIDEO_ID"
   
   # 查看所有缓存状态
   curl "http://127.0.0.1:9009/cache/status"
   ```

3. **重新访问视频页面**
   - 刷新YouTube页面
   - 插件会自动检测并使用缓存的音频文件
   - 跳过下载步骤，直接进行转录翻译

#### 缓存管理
```bash
# 清除指定视频缓存
curl -X DELETE "http://127.0.0.1:9009/cache/clear/VIDEO_ID"

# 查看缓存目录
# 音频缓存: /tmp/yt_cache/audio/
# 字幕缓存: /tmp/yt_cache/subtitles/
```

## 支持的语言

- 简体中文（默认）
- 繁体中文
- 日文
- 韩文  
- English

## 注意事项

### 基本要求
- 需要有效的OpenAI API密钥
- 必须启动后端服务（端口9009）
- 确保网络连接稳定
- Python环境需安装yt-dlp等依赖

### 处理时间
- **首次处理**: 5-15分钟（下载+转录+翻译）
- **缓存命中**: 2-5分钟（仅转录+翻译）
- **处理完成后**: 实时字幕显示，无延迟

### 音频获取策略
1. **自动下载**: 优先下载音频，失败时下载视频提取音频
2. **浏览器Cookie**: 使用Chrome cookies绕过地域限制
3. **手动备用**: 下载失败时支持手动上传音频缓存
4. **智能缓存**: 自动缓存已处理文件，避免重复下载

### 字幕来源说明
- **不依赖YouTube原生字幕**: 始终使用本项目转录技术
- **额外字幕辅助**: 同时下载YouTube字幕用于优化翻译质量
- **双重保障**: 即使视频无原生字幕也能生成高质量双语字幕

## 故障排除

### 常见问题解决

**1. 插件显示"等待后端完成音频下载和翻译..."**
```bash
# 检查后端服务状态
curl http://127.0.0.1:9009/health

# 查看具体错误（替换JOB_ID）
curl http://127.0.0.1:9009/jobs/JOB_ID/state
```

**2. 音频下载失败**
```bash
# 手动下载音频（推荐方法）
yt-dlp --extract-audio --audio-format m4a --cookies-from-browser chrome \
  --geo-bypass "https://www.youtube.com/watch?v=VIDEO_ID"

# 上传到缓存
curl -X POST "http://127.0.0.1:9009/cache/upload_audio/VIDEO_ID" \
  -F "file=@downloaded_audio.m4a"
```

**3. 后端连接失败**
- 确保后端服务已启动：`uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009 --reload`
- 检查端口9009是否被占用：`lsof -i :9009`
- 验证API密钥配置：检查.env文件或环境变量

**4. 翻译质量问题**
- 优先使用gpt-4o模型（质量更高）
- 检查API密钥余额和调用限制
- 查看插件调试日志：按Ctrl+L导出日志

**5. 视频特定问题**
```bash
# 测试特定视频的处理能力
curl -X POST "http://127.0.0.1:9009/verify_ytdlp" \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "YOUR_VIDEO_URL"}'
```

## 技术说明

### 与主项目的关系

本Chrome插件是 **Subtitle Translator** 项目的浏览器扩展版本，**完全复用了主项目的核心转录和翻译技术**：

#### 复用的核心技术
- **转录引擎**: 使用相同的 Parakeet MLX 转录模型和音频处理算法
- **翻译架构**: 三阶段翻译流程（断句→总结→批量翻译）
- **质量保证**: 相同的提示词、反思模式和优化策略

#### Chrome插件的独有优势
- **实时浏览器集成**: 自动检测YouTube视频，无需手动文件操作
- **智能缓存系统**: 音频文件自动缓存，支持手动上传备用
- **用户友好界面**: 调试面板、进度显示、错误反馈

#### 使用建议
- **本地文件处理**: 推荐使用命令行工具 `translate` 和 `transcribe`
- **YouTube视频**: 推荐使用Chrome插件获得最佳体验
- **批量处理**: 使用命令行工具的批处理功能

### 文件说明

- `manifest.json`: 插件配置文件
- `content.js`: 核心功能脚本，处理字幕获取和翻译
- `popup.html/js`: 设置界面
- `style.css`: 双语字幕显示样式

### 工作原理

1. 通过DOM监听检测YouTube字幕变化
2. 将字幕文本发送到OpenAI API翻译
3. 在视频播放器上叠加显示双语字幕

## 版本历史

- v1.0.0: 初始版本，基本的双语字幕功能