# Chrome插件快速测试指南

## 🚀 5分钟快速测试

### 前提条件确认
- ✅ 已配置OpenAI API密钥
- ✅ Python环境准备就绪
- ✅ Chrome浏览器可用

### 步骤1: 启动后端服务（1分钟）

```bash
# 1. 进入项目目录
cd Subtitle-Translator

# 2. 启动服务
uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009 --reload

# 3. 验证服务（新终端窗口）
curl http://127.0.0.1:9009/health
```

**期望输出**:
```json
{
  "ok": true,
  "python": "3.13.3",
  "fastapi": "0.116.1",
  "uvicorn": "0.35.0",  
  "yt_dlp": "2025.08.11"
}
```

### 步骤2: 安装Chrome插件（1分钟）

1. 打开Chrome浏览器
2. 访问 `chrome://extensions/`
3. 开启"开发者模式"
4. 点击"加载已解压的扩展程序"
5. 选择 `chrome-extension` 文件夹 (生产环境)

### 步骤3: 配置插件（1分钟）

1. 点击插件图标
2. 填入配置：
   ```
   API Base URL: https://api.openai.com/v1
   API Key: [你的API密钥]
   目标语言: 简体中文
   模型: gpt-4o-mini
   ```
3. 点击"保存设置"

### 步骤4: 测试视频处理（2分钟）

1. 打开测试视频：
   ```
   https://www.youtube.com/watch?v=8KkKuTCFvzI
   ```

2. 观察插件状态：
   - 右上角应显示调试面板
   - 状态从"等待后端"→"下载音频"→"转录翻译"

3. 等待处理完成（约5-15分钟首次处理）

4. 验证双语字幕显示

## 🔧 快速故障排除

### 问题1: 后端服务启动失败

```bash
# 检查端口占用
lsof -i :9009

# 杀死占用进程
kill -9 <PID>

# 安装缺失依赖
uv add python-multipart fastapi uvicorn

# 重新启动
uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009 --reload
```

### 问题2: 插件无响应

```bash
# 检查服务连接
curl http://127.0.0.1:9009/health

# 查看Chrome控制台
# F12 → Console 查看错误信息

# 导出插件日志
# 在YouTube页面按 Ctrl+L
```

### 问题3: 音频下载失败

```bash
# 手动下载测试
uv run python -m yt_dlp --extract-audio --audio-format m4a \
  --cookies-from-browser chrome --geo-bypass \
  "https://www.youtube.com/watch?v=8KkKuTCFvzI"

# 检查yt-dlp版本
uv run python -m yt_dlp --version

# 更新yt-dlp
uv add yt-dlp@latest
```

## 📊 测试验证清单

### 后端服务验证
- [ ] `/health` 端点返回正常
- [ ] `/cache/status` 返回缓存状态
- [ ] `/test_download` 测试下载功能

```bash
# 全面测试
curl http://127.0.0.1:9009/health
curl http://127.0.0.1:9009/cache/status  
curl -X POST http://127.0.0.1:9009/test_download
```

### Chrome插件验证
- [ ] 插件成功加载到Chrome
- [ ] 配置保存成功
- [ ] YouTube页面显示调试面板
- [ ] 插件能连接后端服务

### 处理流程验证
- [ ] 视频ID正确提取
- [ ] 后端任务提交成功
- [ ] 音频下载或缓存使用
- [ ] 转录翻译完成
- [ ] 双语字幕正确显示

## 🎯 推荐测试视频

### 短视频（快速测试）
- **TED Talk短片**: `https://www.youtube.com/watch?v=8KkKuTCFvzI`
  - 时长: ~3分钟
  - 语音清晰，适合测试转录质量

### 中等视频（完整测试）  
- **技术讲座**: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
  - 时长: ~4分钟
  - 经典测试视频，稳定可靠

### 测试不同场景
- **有原生字幕**: 测试字幕辅助优化
- **无原生字幕**: 测试纯转录能力
- **不同语速**: 测试转录适应性
- **技术内容**: 测试术语翻译质量

## 📋 测试报告模板

```
# Chrome插件测试报告

## 测试环境
- 操作系统: 
- Python版本: 
- Chrome版本:
- API服务商:

## 测试视频
- URL: 
- 时长:
- 有无原生字幕:

## 测试结果
- [ ] 后端服务启动 (耗时: _分钟)
- [ ] 插件配置完成
- [ ] 视频处理启动
- [ ] 音频下载/缓存 (耗时: _分钟)
- [ ] 转录完成 (耗时: _分钟)  
- [ ] 翻译完成 (耗时: _分钟)
- [ ] 字幕显示正常

## 性能数据
- 总处理时间: _分钟
- 音频文件大小: _MB
- 生成字幕数量: _条
- 翻译质量评分: _/10

## 遇到的问题
1. 
2. 
3. 

## 改进建议
1.
2. 
3.
```

## 🔄 持续测试建议

### 日常开发测试
```bash
# 每次代码修改后快速验证
curl http://127.0.0.1:9009/health && echo "✅ 后端正常"
```

### 功能回归测试
```bash
# 定期运行完整测试套件
./chrome-extension/test_suite.sh
```

### 性能基准测试
```bash
# 监控处理时间和资源使用
time curl -X POST http://127.0.0.1:9009/translate_youtube \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=TEST_VIDEO"}'
```

通过这个快速测试指南，你应该能在5分钟内完成基本的功能验证！