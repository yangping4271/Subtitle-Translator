import Foundation
import OSLog

// MARK: - 统一日志管理器
class LogManager {
    static let shared = LogManager()
    
    private let subsystem = "com.subtitletranslator.app"
    private let logFileURL: URL
    private let maxLogFileSize: Int = 10 * 1024 * 1024 // 10MB
    private let maxLogFiles: Int = 3
    
    // 不同组件的Logger
    let pythonBridge: Logger
    let processingManager: Logger
    let configuration: Logger
    let ui: Logger
    let general: Logger
    
    private init() {
        // 设置日志文件目录
        let logsDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("SubtitleTranslator")
            .appendingPathComponent("logs")
        
        try? FileManager.default.createDirectory(at: logsDirectory, withIntermediateDirectories: true)
        
        self.logFileURL = logsDirectory.appendingPathComponent("app.log")
        
        // 初始化各种Logger
        self.pythonBridge = Logger(subsystem: subsystem, category: "PythonBridge")
        self.processingManager = Logger(subsystem: subsystem, category: "ProcessingManager")
        self.configuration = Logger(subsystem: subsystem, category: "Configuration")
        self.ui = Logger(subsystem: subsystem, category: "UI")
        self.general = Logger(subsystem: subsystem, category: "General")
        
        // 清理旧日志文件
        cleanupOldLogs()
        
        // 记录启动日志
        logToFile("应用启动", level: .info, category: "General")
    }
    
    // MARK: - 公共日志方法
    
    func logInfo(_ message: String, category: String = "General") {
        logToFile(message, level: .info, category: category)
        getLogger(for: category).info("\\(message)")
    }
    
    func logError(_ message: String, category: String = "General") {
        logToFile(message, level: .error, category: category)
        getLogger(for: category).error("\\(message)")
    }
    
    func logWarning(_ message: String, category: String = "General") {
        logToFile(message, level: .warning, category: category)
        getLogger(for: category).warning("\\(message)")
    }
    
    func logDebug(_ message: String, category: String = "General") {
        logToFile(message, level: .debug, category: category)
        getLogger(for: category).debug("\\(message)")
    }
    
    // MARK: - 文件日志
    
    private func logToFile(_ message: String, level: LogLevel, category: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let logEntry = "[\\(timestamp)] [\\(level.rawValue.uppercased())] [\\(category)] \\(message)\\n"
        
        guard let data = logEntry.data(using: .utf8) else { return }
        
        // 检查文件大小，如果太大则轮转
        if shouldRotateLog() {
            rotateLogFile()
        }
        
        // 写入日志
        if FileManager.default.fileExists(atPath: logFileURL.path) {
            if let fileHandle = try? FileHandle(forWritingTo: logFileURL) {
                fileHandle.seekToEndOfFile()
                fileHandle.write(data)
                fileHandle.closeFile()
            }
        } else {
            try? data.write(to: logFileURL)
        }
    }
    
    private func shouldRotateLog() -> Bool {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: logFileURL.path),
              let fileSize = attributes[.size] as? Int else {
            return false
        }
        return fileSize > maxLogFileSize
    }
    
    private func rotateLogFile() {
        let logsDirectory = logFileURL.deletingLastPathComponent()
        
        // 移动当前日志文件
        for i in (1..<maxLogFiles).reversed() {
            let oldFile = logsDirectory.appendingPathComponent("app.\\(i).log")
            let newFile = logsDirectory.appendingPathComponent("app.\\(i + 1).log")
            
            if FileManager.default.fileExists(atPath: oldFile.path) {
                try? FileManager.default.moveItem(at: oldFile, to: newFile)
            }
        }
        
        // 移动当前文件到 .1
        let rotatedFile = logsDirectory.appendingPathComponent("app.1.log")
        try? FileManager.default.moveItem(at: logFileURL, to: rotatedFile)
        
        // 删除最老的文件
        let oldestFile = logsDirectory.appendingPathComponent("app.\\(maxLogFiles).log")
        try? FileManager.default.removeItem(at: oldestFile)
    }
    
    private func cleanupOldLogs() {
        let logsDirectory = logFileURL.deletingLastPathComponent()
        
        // 清理超过maxLogFiles数量的日志文件
        for i in (maxLogFiles + 1)...10 {
            let oldFile = logsDirectory.appendingPathComponent("app.\\(i).log")
            if FileManager.default.fileExists(atPath: oldFile.path) {
                try? FileManager.default.removeItem(at: oldFile)
            }
        }
    }
    
    // MARK: - 日志查看
    
    func getAllLogs() -> String {
        var allLogs = ""
        
        // 读取当前日志文件
        if let currentLog = try? String(contentsOf: logFileURL, encoding: .utf8) {
            allLogs = currentLog
        }
        
        // 读取轮转的日志文件
        let logsDirectory = logFileURL.deletingLastPathComponent()
        for i in 1..<maxLogFiles {
            let rotatedFile = logsDirectory.appendingPathComponent("app.\\(i).log")
            if let rotatedLog = try? String(contentsOf: rotatedFile, encoding: .utf8) {
                allLogs = rotatedLog + "\\n=== 轮转日志分隔线 ===\\n" + allLogs
            }
        }
        
        return allLogs.isEmpty ? "暂无日志" : allLogs
    }
    
    func getLogFileURL() -> URL {
        return logFileURL
    }
    
    func getLogDirectory() -> URL {
        return logFileURL.deletingLastPathComponent()
    }
    
    // MARK: - 私有方法
    
    private func getLogger(for category: String) -> Logger {
        switch category.lowercased() {
        case "pythonbridge":
            return pythonBridge
        case "processingmanager":
            return processingManager
        case "configuration":
            return configuration
        case "ui":
            return ui
        default:
            return general
        }
    }
}

// MARK: - 日志级别
enum LogLevel: String, CaseIterable {
    case debug = "debug"
    case info = "info"
    case warning = "warning"
    case error = "error"
    
    var emoji: String {
        switch self {
        case .debug:
            return "🔍"
        case .info:
            return "ℹ️"
        case .warning:
            return "⚠️"
        case .error:
            return "❌"
        }
    }
}

// MARK: - 日志过滤器
struct LogFilter {
    var levels: Set<LogLevel> = Set(LogLevel.allCases)
    var categories: Set<String> = []
    var timeRange: DateInterval?
    var searchText: String = ""
    
    func matches(_ logEntry: LogEntry) -> Bool {
        // 检查级别
        if !levels.contains(logEntry.level) {
            return false
        }
        
        // 检查分类
        if !categories.isEmpty && !categories.contains(logEntry.category) {
            return false
        }
        
        // 检查时间范围
        if let timeRange = timeRange, !timeRange.contains(logEntry.timestamp) {
            return false
        }
        
        // 检查搜索文本
        if !searchText.isEmpty && !logEntry.message.localizedCaseInsensitiveContains(searchText) {
            return false
        }
        
        return true
    }
}

// MARK: - 日志条目
struct LogEntry {
    let timestamp: Date
    let level: LogLevel
    let category: String
    let message: String
    
    var formattedString: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return "[\\(formatter.string(from: timestamp))] [\\(level.emoji)] [\\(category)] \\(message)"
    }
}

// MARK: - 便利方法扩展
extension LogManager {
    func exportLogs() -> URL? {
        let allLogs = getAllLogs()
        
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let filename = "SubtitleTranslator_logs_\\(formatter.string(from: Date())).txt"
        
        let exportURL = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        
        do {
            try allLogs.write(to: exportURL, atomically: true, encoding: .utf8)
            return exportURL
        } catch {
            logError("导出日志失败: \\(error.localizedDescription)")
            return nil
        }
    }
    
    func clearLogs() {
        // 删除所有日志文件
        let logsDirectory = logFileURL.deletingLastPathComponent()
        
        try? FileManager.default.removeItem(at: logFileURL)
        
        for i in 1..<maxLogFiles {
            let rotatedFile = logsDirectory.appendingPathComponent("app.\\(i).log")
            try? FileManager.default.removeItem(at: rotatedFile)
        }
        
        logInfo("日志已清空")
    }
}