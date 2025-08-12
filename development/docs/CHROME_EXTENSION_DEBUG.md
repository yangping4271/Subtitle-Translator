# Chrome插件调试指南 - 解决"等待后端完成音频下载和翻译"问题

## 问题描述

Chrome插件显示"等待后端完成音频下载和翻译..."状态，但一直无法完成处理。

## 解决方案

### 1. 启动后端服务

首先确保后端服务正常运行：

```bash
# 方式1: 使用uvicorn启动（推荐）
uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009 --reload

# 方式2: 直接启动Python脚本
uv run python backend/server.py
```

### 2. 检查服务状态

访问 http://127.0.0.1:9009/health 确认服务正常

### 3. 检查依赖安装

确保以下依赖已安装：

```bash
pip install fastapi uvicorn yt-dlp
```

### 4. 检查API配置

确保已配置OpenAI API密钥：

```bash
# 在.env文件中添加
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# 或设置环境变量
export OPENAI_API_KEY=your_api_key_here
```

### 5. 测试后端功能

```bash
# 测试健康检查
curl http://127.0.0.1:9009/health

# 测试下载功能
curl -X POST http://127.0.0.1:9009/test_download
```

## 调试步骤

### 步骤1: 检查后端日志

启动后端服务时观察控制台输出，查看是否有错误信息。

### 步骤2: 使用Chrome开发者工具

1. 打开YouTube页面
2. 按F12打开开发者工具
3. 查看Console面板的错误信息
4. 查看Network面板的请求状态

### 步骤3: 导出插件日志

在YouTube页面按 `Ctrl+L` 导出详细的调试日志。

### 步骤4: 检查网络连接

确保：
- 本地端口9009未被占用
- 防火墙允许访问
- Chrome没有阻止跨域请求

## 常见问题及解决方案

### 问题1: "后端连接失败"

**原因**: 后端服务未启动或端口被占用

**解决方案**:
```bash
# 检查端口占用
lsof -i :9009

# 杀死占用进程
kill -9 <PID>

# 重新启动服务
uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009 --reload
```

### 问题2: "yt-dlp下载失败"

**原因**: YouTube访问限制或yt-dlp版本过旧

**解决方案**:
```bash
# 更新yt-dlp
pip install --upgrade yt-dlp

# 检查版本
yt-dlp --version
```

### 问题3: "subtitle_translator不可用"

**原因**: 主项目未正确安装

**解决方案**:
```bash
# 在项目根目录安装
uv tool install .

# 或开发模式
uv sync --dev
```

### 问题4: "翻译API调用失败"

**原因**: API密钥配置错误或网络问题

**解决方案**:
1. 检查API密钥是否正确
2. 确认网络可以访问OpenAI API
3. 尝试使用代理或其他API端点

## 增强调试功能

### 后端改进 (已修复)

1. **详细的进度追踪**: 添加了stage和message字段
2. **错误信息增强**: 包含完整的traceback信息
3. **超时机制**: 前端10分钟超时保护
4. **健康检查**: 自动测试后端连接和下载功能

### 前端改进 (已修复)

1. **状态显示优化**: 实时显示处理阶段
2. **错误处理增强**: 详细记录和显示错误信息
3. **轮询优化**: 降低轮询频率(2秒)，减少服务器压力
4. **自动重试**: 智能重试机制

## 监控和维护

### 查看服务状态

```bash
# 检查服务进程
ps aux | grep server.py

# 查看端口监听
netstat -tulpn | grep 9009
```

### 日志管理

- 后端日志：控制台输出
- 前端日志：Chrome开发者工具Console + Ctrl+L导出

### 性能优化建议

1. 使用SSD存储临时文件
2. 确保网络连接稳定
3. 定期清理临时文件
4. 监控内存使用情况

## 故障排除清单

- [ ] 后端服务正常启动 (http://127.0.0.1:9009/health)
- [ ] yt-dlp版本最新且可用
- [ ] OpenAI API密钥配置正确
- [ ] Chrome允许跨域请求
- [ ] 防火墙端口开放
- [ ] 网络连接稳定
- [ ] 磁盘空间充足 (/tmp目录)
- [ ] Python依赖完整安装

遵循这个调试指南，应该能够解决大部分Chrome插件等待后端的问题。如果问题依然存在，请收集详细的日志信息进行进一步分析。