# YouTube双语字幕翻译器 - 后端服务

这是YouTube双语字幕翻译器的后端转录服务，基于FastAPI构建，提供实时音频转录和翻译功能。

## 功能特性

- 🎵 **音频转录**: 使用Parakeet MLX模型进行高质量音频转录
- 🌐 **实时翻译**: 集成OpenAI等LLM提供双语字幕翻译
- 📦 **任务队列**: 支持后台处理长视频转录任务
- 💾 **智能缓存**: 自动缓存转录结果，避免重复处理
- 🔄 **增量更新**: 支持实时字幕流式输出
- 🎯 **Chrome扩展集成**: 专为YouTube Chrome扩展优化

## 快速启动

### 1. 安装依赖
```bash
# 确保在项目根目录
cd /Users/yangping/Subtitle-Translator

# 安装Python依赖
uv sync --dev
```

### 2. 启动服务
```bash
# 方式1: 使用uvicorn直接启动（推荐）
uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009 --reload

# 方式2: 使用Python模块启动
uv run python backend/server.py
```

### 3. 验证服务
```bash
# 检查服务健康状态
curl http://127.0.0.1:9009/health

# 期望返回:
# {"status":"ok","python":"3.13","jobs_count":0,"cache_entries":0}
```

## API 端点

### 核心端点
- `GET /health` - 服务健康检查
- `POST /transcribe` - 音频转录请求
- `GET /job/{job_id}` - 查询任务状态
- `GET /job/{job_id}/srt` - 获取SRT字幕文件
- `GET /job/{job_id}/events` - 获取实时转录事件

### 管理端点
- `GET /jobs` - 查看所有任务
- `DELETE /job/{job_id}` - 删除指定任务
- `POST /clear_cache` - 清理缓存

## 服务配置

### 环境变量
```bash
# OpenAI API配置（用于翻译）
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# 模型配置
TRANSCRIPTION_MODEL=mlx-community/parakeet-ctc-0.6b-no-fp16
TRANSLATION_MODEL=gpt-4o-mini

# 服务配置
HOST=127.0.0.1
PORT=9009
```

### 默认设置
- **监听地址**: `0.0.0.0:9009`
- **工作模式**: 开发模式（支持热重载）
- **缓存策略**: 内存缓存 + 文件缓存
- **最大并发**: 基于系统资源自动调节

## 使用示例

### 1. 转录YouTube视频
```bash
curl -X POST "http://127.0.0.1:9009/transcribe" \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "dQw4w9WgXcQ",
    "language": "en",
    "translate_to": "zh"
  }'
```

### 2. 查询任务状态
```bash
curl "http://127.0.0.1:9009/job/your-job-id"
```

### 3. 获取SRT字幕
```bash
# 英文字幕
curl "http://127.0.0.1:9009/job/your-job-id/srt?lang=en"

# 中文翻译
curl "http://127.0.0.1:9009/job/your-job-id/srt?lang=zh"
```

## Chrome扩展集成

### 连接配置
Chrome扩展通过以下方式连接后端：
```javascript
// 默认后端地址
const backendUrl = 'http://127.0.0.1:9009';

// 健康检查
const health = await fetch(`${backendUrl}/health`);
```

### 工作流程
1. 扩展检测YouTube视频播放
2. 提取视频ID并发送转录请求
3. 后端下载音频并进行转录
4. 实时推送转录结果给扩展
5. 扩展调用翻译API获取双语字幕
6. 在视频上显示双语字幕

## 故障排除

### 1. 端口占用
```bash
# 检查端口占用
lsof -i :9009

# 杀掉占用进程
kill <PID>
```

### 2. 依赖问题
```bash
# 重新安装依赖
uv sync --dev

# 检查MLX安装（仅限Apple Silicon Mac）
uv run python -c "import mlx.core; print('MLX OK')"
```

### 3. 模型下载
```bash
# 预下载转录模型
uv run python -m subtitle_translator.transcription_core.cli model download mlx-community/parakeet-ctc-0.6b-no-fp16
```

### 4. 常见错误

**426 Upgrade Required**
- **原因**: Chrome扩展错误地通过代理访问本地服务
- **解决**: 确保扩展中的`needsProxy`函数不包含`127.0.0.1`和`localhost`
- **检查**: `chrome-extension/content-with-logger.js`和`translation-processor.js`中的代理逻辑

**404 Not Found**
- **原因**: 服务未启动或端口错误
- **解决**: 确认服务已启动且监听正确端口
- **检查**: `lsof -i :9009`查看端口占用

**500 Internal Error**
- **原因**: 模型加载失败或依赖问题
- **解决**: 检查MLX模型是否正确安装和加载
- **检查**: 服务器日志中的详细错误信息

**CORS错误**
- **原因**: 跨域请求被浏览器阻止
- **解决**: 确保本地服务允许来自Chrome扩展的请求
- **注意**: 本地服务通常不需要代理

## 开发说明

### 项目结构
```
backend/
├── server.py          # FastAPI主服务器
├── models/            # 转录模型管理
├── cache/             # 缓存系统
├── utils/             # 工具函数
└── README.md          # 本文档
```

### 添加新功能
1. 在`server.py`中定义新的API端点
2. 实现相应的业务逻辑
3. 更新Chrome扩展的调用代码
4. 测试端到端功能

### 性能优化
- 模型加载缓存：避免重复加载MLX模型
- 音频分块处理：支持长视频转录
- 并发任务管理：合理控制系统资源使用
- 智能缓存策略：基于视频ID和内容Hash

## API文档

启动服务后，访问以下地址查看完整API文档：
- Swagger UI: http://127.0.0.1:9009/docs
- ReDoc: http://127.0.0.1:9009/redoc

## 许可证

本项目遵循与主项目相同的许可证。