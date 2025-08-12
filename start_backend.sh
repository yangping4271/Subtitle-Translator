#!/bin/bash

# YouTube双语字幕翻译后端服务启动脚本
# 自动处理端口冲突，支持启动/重启/状态检查

PORT=9009
SERVICE_NAME="YouTube双语字幕翻译服务"
UVICORN_CMD="uv run uvicorn backend.server:app --host 0.0.0.0 --port $PORT --reload"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查端口是否被占用
check_port() {
    local port=$1
    lsof -i :$port 2>/dev/null
}

# 获取占用端口的进程PID
get_port_pid() {
    local port=$1
    lsof -ti :$port 2>/dev/null
}

# 杀死占用端口的进程
kill_port_process() {
    local port=$1
    local pids=$(get_port_pid $port)
    
    if [ -n "$pids" ]; then
        print_warning "发现端口 $port 被以下进程占用:"
        lsof -i :$port
        echo
        
        for pid in $pids; do
            local process_info=$(ps -p $pid -o pid,comm,args --no-headers 2>/dev/null)
            if [ -n "$process_info" ]; then
                print_warning "正在终止进程: $process_info"
                kill $pid 2>/dev/null
                sleep 1
                
                # 如果进程仍然存在，强制杀死
                if kill -0 $pid 2>/dev/null; then
                    print_warning "强制终止进程 $pid"
                    kill -9 $pid 2>/dev/null
                fi
            fi
        done
        
        # 等待端口释放
        local retry_count=0
        while [ $retry_count -lt 5 ] && [ -n "$(get_port_pid $port)" ]; do
            print_status "等待端口 $port 释放... ($((retry_count + 1))/5)"
            sleep 1
            retry_count=$((retry_count + 1))
        done
        
        if [ -n "$(get_port_pid $port)" ]; then
            print_error "无法释放端口 $port，请手动检查"
            return 1
        else
            print_success "端口 $port 已释放"
        fi
    fi
    return 0
}

# 检查uvicorn是否在运行
check_uvicorn_status() {
    local uvicorn_pids=$(pgrep -f "uvicorn.*backend\.server")
    if [ -n "$uvicorn_pids" ]; then
        print_status "发现运行中的uvicorn进程:"
        for pid in $uvicorn_pids; do
            ps -p $pid -o pid,ppid,etime,comm,args --no-headers 2>/dev/null
        done
        return 0
    else
        return 1
    fi
}

# 启动服务
start_service() {
    print_status "启动 $SERVICE_NAME..."
    print_status "执行命令: $UVICORN_CMD"
    echo
    
    # 检查是否在项目目录中
    if [ ! -f "backend/server.py" ]; then
        print_error "未找到 backend/server.py 文件"
        print_error "请确保在项目根目录下运行此脚本"
        exit 1
    fi
    
    # 启动服务
    exec $UVICORN_CMD
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  start     启动服务（默认行为）"
    echo "  restart   重启服务"
    echo "  stop      停止服务"
    echo "  status    查看服务状态"
    echo "  help      显示帮助信息"
    echo
    echo "功能说明:"
    echo "  - 自动检查并处理端口冲突"
    echo "  - 智能重启已运行的服务"
    echo "  - 彩色输出和详细状态信息"
    echo "  - 优雅关闭和强制终止支持"
}

# 停止服务
stop_service() {
    print_status "停止 $SERVICE_NAME..."
    
    # 停止uvicorn进程
    local uvicorn_pids=$(pgrep -f "uvicorn.*backend\.server")
    if [ -n "$uvicorn_pids" ]; then
        for pid in $uvicorn_pids; do
            print_warning "停止uvicorn进程 $pid"
            kill $pid 2>/dev/null
        done
        sleep 2
    fi
    
    # 清理端口
    kill_port_process $PORT
    
    print_success "服务已停止"
}

# 显示服务状态
show_status() {
    echo "=== $SERVICE_NAME 状态 ==="
    echo
    
    # 检查端口状态
    local port_info=$(check_port $PORT)
    if [ -n "$port_info" ]; then
        print_status "端口 $PORT 状态: 被占用"
        echo "$port_info"
    else
        print_status "端口 $PORT 状态: 空闲"
    fi
    
    echo
    
    # 检查uvicorn进程
    if check_uvicorn_status; then
        print_success "服务状态: 运行中"
    else
        print_warning "服务状态: 未运行"
    fi
    
    echo
    
    # 检查服务可访问性
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        print_success "服务健康检查: 通过"
        print_status "访问地址: http://127.0.0.1:$PORT"
        print_status "API文档: http://127.0.0.1:$PORT/docs"
    else
        print_warning "服务健康检查: 失败"
    fi
}

# 主逻辑
main() {
    local action="${1:-start}"
    
    print_status "=== $SERVICE_NAME 管理脚本 ==="
    print_status "目标端口: $PORT"
    echo
    
    case $action in
        "start")
            # 检查是否已有服务运行
            if check_uvicorn_status; then
                print_warning "检测到uvicorn服务已在运行"
                read -p "是否重启服务? [y/N]: " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    stop_service
                    echo
                    start_service
                else
                    print_status "保持现有服务运行"
                    show_status
                fi
            else
                # 清理端口并启动
                kill_port_process $PORT
                echo
                start_service
            fi
            ;;
        "restart")
            stop_service
            echo
            start_service
            ;;
        "stop")
            stop_service
            ;;
        "status")
            show_status
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "未知选项: $action"
            echo
            show_help
            exit 1
            ;;
    esac
}

# 脚本入口
main "$@"