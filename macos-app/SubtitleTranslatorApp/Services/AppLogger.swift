import Foundation
import OSLog

class AppLogger {
    static let shared = AppLogger()
    private let logFileURL: URL
    private let logger = Logger(subsystem: "com.subtitletranslator.app", category: "App")
    private let dateFormatter: ISO8601DateFormatter
    private let queue = DispatchQueue(label: "com.subtitletranslator.applogger", qos: .background)
    
    private init() {
        // 将日志文件放在macos-app/logs目录下
        let projectRoot = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        
        let logsDirectory = projectRoot.appendingPathComponent("logs")
        
        // 确保logs目录存在
        try? FileManager.default.createDirectory(at: logsDirectory, withIntermediateDirectories: true)
        
        self.logFileURL = logsDirectory.appendingPathComponent("app.log")
        self.dateFormatter = ISO8601DateFormatter()
        
        // 创建或清空日志文件
        createLogFile()
    }
    
    private func createLogFile() {
        let header = """
        ===== Subtitle Translator App Log =====
        Started at: \(dateFormatter.string(from: Date()))
        =====================================
        
        """
        
        do {
            try header.write(to: logFileURL, atomically: true, encoding: .utf8)
        } catch {
            print("Failed to create log file: \(error)")
        }
    }
    
    func log(_ message: String, level: LogLevel = .info, file: String = #file, function: String = #function, line: Int = #line) {
        let filename = URL(fileURLWithPath: file).lastPathComponent
        let timestamp = dateFormatter.string(from: Date())
        let logEntry = "[\(timestamp)] [\(level.rawValue)] [\(filename):\(line)] \(function) - \(message)\n"
        
        // 输出到控制台
        print(logEntry.trimmingCharacters(in: .newlines))
        
        // 输出到系统日志
        switch level {
        case .debug:
            logger.debug("\(message)")
        case .info:
            logger.info("\(message)")
        case .warning:
            logger.warning("\(message)")
        case .error:
            logger.error("\(message)")
        }
        
        // 异步写入文件
        queue.async { [weak self] in
            guard let self = self else { return }
            
            if let data = logEntry.data(using: .utf8) {
                if FileManager.default.fileExists(atPath: self.logFileURL.path) {
                    if let fileHandle = try? FileHandle(forWritingTo: self.logFileURL) {
                        fileHandle.seekToEndOfFile()
                        fileHandle.write(data)
                        fileHandle.closeFile()
                    }
                } else {
                    try? data.write(to: self.logFileURL)
                }
            }
        }
    }
    
    func debug(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .debug, file: file, function: function, line: line)
    }
    
    func info(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .info, file: file, function: function, line: line)
    }
    
    func warning(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .warning, file: file, function: function, line: line)
    }
    
    func error(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .error, file: file, function: function, line: line)
    }
    
    enum LogLevel: String {
        case debug = "DEBUG"
        case info = "INFO"
        case warning = "WARNING"
        case error = "ERROR"
    }
}

// 全局便捷函数
func appLog(_ message: String, level: AppLogger.LogLevel = .info, file: String = #file, function: String = #function, line: Int = #line) {
    AppLogger.shared.log(message, level: level, file: file, function: function, line: line)
}