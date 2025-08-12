// 日志收集器 - 可以将Console日志保存到本地文件
class DebugLogger {
  constructor() {
    this.logs = [];
    this.maxLogs = 1000; // 最多保存1000条日志
  }

  log(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level,
      message,
      data: data ? JSON.stringify(data, null, 2) : null
    };
    
    this.logs.push(logEntry);
    
    // 保持日志数量限制
    if (this.logs.length > this.maxLogs) {
      this.logs.shift(); // 删除最旧的日志
    }
    
    // 同时输出到控制台
    const consoleMessage = `[${level}] ${message}`;
    if (data) {
      console[level.toLowerCase()](consoleMessage, data);
    } else {
      console[level.toLowerCase()](consoleMessage);
    }
  }

  info(message, data) { this.log('INFO', message, data); }
  warn(message, data) { this.log('WARN', message, data); }
  error(message, data) { this.log('ERROR', message, data); }
  debug(message, data) { this.log('DEBUG', message, data); }

  // 导出日志到文件
  exportLogs() {
    const logText = this.logs.map(log => {
      let line = `[${log.timestamp}] [${log.level}] ${log.message}`;
      if (log.data) {
        line += `\n${log.data}`;
      }
      return line;
    }).join('\n\n');

    // 创建下载链接
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `youtube-subtitle-debug-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.log`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    this.info('日志已导出到下载文件夹');
  }

  // 获取最近的日志（用于显示）
  getRecentLogs(count = 10) {
    return this.logs.slice(-count);
  }

  // 清空日志
  clearLogs() {
    this.logs = [];
    this.info('日志已清空');
  }
}

// 创建全局日志器
window.debugLogger = new DebugLogger();