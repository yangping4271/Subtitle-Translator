# 后端服务启动脚本使用指南

## 脚本文件

### 1. 完整功能脚本: `start_backend.sh`
智能后端服务管理脚本，支持完整的生命周期管理。

#### 使用方法:
```bash
# 启动服务（默认）
./start_backend.sh
./start_backend.sh start

# 重启服务
./start_backend.sh restart

# 停止服务
./start_backend.sh stop

# 查看状态
./start_backend.sh status

# 显示帮助
./start_backend.sh help
```

#### 功能特点:
- ✅ **智能端口管理**: 自动检测和清理端口冲突
- ✅ **优雅重启**: 已运行服务会提示是否重启
- ✅ **进程监控**: 实时显示uvicorn进程状态
- ✅ **健康检查**: 验证服务可访问性
- ✅ **彩色输出**: 清晰的状态提示
- ✅ **错误处理**: 完善的异常情况处理

### 2. 快速启动脚本: `quick_start.sh`
一键启动脚本，适合日常快速使用。

```bash
# 一键启动/重启
./quick_start.sh
```

## 使用场景

### 场景1: 首次启动
```bash
./start_backend.sh
# 或
./quick_start.sh
```

### 场景2: 端口被占用时
脚本会自动处理：
1. 检测端口占用情况
2. 显示占用进程信息
3. 优雅终止冲突进程
4. 启动新的服务

### 场景3: 开发调试
```bash
# 查看当前状态
./start_backend.sh status

# 重启服务应用代码变更
./start_backend.sh restart
```

### 场景4: 服务维护
```bash
# 停止服务
./start_backend.sh stop

# 检查是否完全停止
./start_backend.sh status
```

## 输出示例

### 正常启动输出:
```
[INFO] === YouTube双语字幕翻译服务 管理脚本 ===
[INFO] 目标端口: 9009

[INFO] 启动 YouTube双语字幕翻译服务...
[INFO] 执行命令: uv run uvicorn backend.server:app --host 0.0.0.0 --port 9009 --reload

INFO: Will watch for changes in these directories: ['/Users/xxx/Subtitle-Translator']
INFO: Uvicorn running on http://0.0.0.0:9009 (Press CTRL+C to quit)
```

### 端口冲突处理输出:
```
[WARNING] 发现端口 9009 被以下进程占用:
COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
node    12345 user   12u  IPv4 ...      0t0  TCP *:9009 (LISTEN)

[WARNING] 正在终止进程: 12345 node http-server
[SUCCESS] 端口 9009 已释放
```

### 状态检查输出:
```
=== YouTube双语字幕翻译服务 状态 ===

[INFO] 端口 9009 状态: 被占用
[SUCCESS] 服务状态: 运行中
[SUCCESS] 服务健康检查: 通过
[INFO] 访问地址: http://127.0.0.1:9009
[INFO] API文档: http://127.0.0.1:9009/docs
```

## 故障排除

### 1. 脚本权限问题
```bash
chmod +x start_backend.sh quick_start.sh
```

### 2. 项目路径问题
确保在项目根目录（包含backend/server.py）运行脚本。

### 3. 环境依赖问题
确保已安装项目依赖：
```bash
uv sync --dev
```

### 4. 端口持续被占用
手动检查并清理：
```bash
lsof -i :9009
kill -9 <PID>
```

## 服务访问

启动成功后，可通过以下地址访问：

- **API服务**: http://127.0.0.1:9009
- **健康检查**: http://127.0.0.1:9009/health  
- **API文档**: http://127.0.0.1:9009/docs
- **配置信息**: http://127.0.0.1:9009/config

## 建议用法

### 日常开发
```bash
# 启动开发环境
./quick_start.sh

# 检查服务状态
./start_backend.sh status
```

### 生产部署
```bash
# 完整启动流程
./start_backend.sh start

# 监控服务状态
./start_backend.sh status
```

脚本设计确保了服务的稳定性和易用性，无论是端口冲突还是进程残留，都能自动处理并提供清晰的状态反馈。