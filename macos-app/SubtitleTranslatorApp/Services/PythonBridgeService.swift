import Foundation
import OSLog

class PythonBridgeService {
    private let cliPath: String
    private let uvExecutable = "/opt/homebrew/bin/uv"
    private let logger = Logger(subsystem: "com.subtitletranslator.app", category: "PythonBridge")
    private let logFileURL: URL
    
    init() {
        self.cliPath = PythonBridgeService.findCLIPath()
        
        // 使用项目结构中的logs目录
        let projectRoot = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        
        let logsDirectory = projectRoot.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: logsDirectory, withIntermediateDirectories: true)
        self.logFileURL = logsDirectory.appendingPathComponent("python_bridge.log")
        
        logMessage("=== PythonBridgeService 初始化 ===")
        logMessage("CLI路径: \(self.cliPath)")
        logMessage("CLI目录存在: \(FileManager.default.fileExists(atPath: self.cliPath))")
        
        let uvPath = findUVPath()
        logMessage("uv路径: \(uvPath)")
        logMessage("uv存在: \(FileManager.default.fileExists(atPath: uvPath))")
        
        // 异步测试CLI可用性
        Task {
            await testCLIAvailability()
        }
    }
    
    private static func findCLIPath() -> String {
        if let bundlePath = Bundle.main.resourcePath {
            let cliPath = "\(bundlePath)/cli"
            if FileManager.default.fileExists(atPath: cliPath) {
                return cliPath
            }
        }
        
        let appPath = Bundle.main.bundlePath
        let appParentPath = (appPath as NSString).deletingLastPathComponent
        let devPath = "\(appParentPath)/cli"
        
        if FileManager.default.fileExists(atPath: devPath) {
            return devPath
        }
        
        return "/usr/local/lib/subtitle-translator/cli"
    }
    
    private func testCLIAvailability() async {
        logMessage("=== 测试CLI可用性 ===")
        
        do {
            // 首先测试帮助命令
            let testOutput = try await executeCommand(
                arguments: ["run", "python", "-m", "subtitle_translator.cli", "--help"],
                workingDirectory: cliPath
            )
            
            if testOutput.contains("usage:") || testOutput.contains("translate") {
                logMessage("✅ CLI帮助测试成功")
                
                // 测试实际导入是否正常工作
                let importTest = try await executeCommand(
                    arguments: ["run", "python", "-c", "import typer; import mlx; print('Dependencies OK')"],
                    workingDirectory: cliPath
                )
                
                if importTest.contains("Dependencies OK") {
                    logMessage("✅ 依赖包测试成功 - 所有必需的模块都可以导入")
                } else {
                    logMessage("⚠️ 依赖包测试警告 - 某些模块可能缺失")
                    // 尝试同步依赖
                    await syncDependencies()
                }
            } else {
                logMessage("⚠️ CLI测试警告 - 输出不符合预期: \(String(testOutput.prefix(200)))")
            }
        } catch {
            logMessage("❌ CLI测试失败: \(error.localizedDescription)")
            logMessage("建议：请检查CLI是否正确安装，或在终端中手动测试")
            // 尝试同步依赖
            await syncDependencies()
        }
    }
    
    private func syncDependencies() async {
        logMessage("=== 尝试同步依赖 ===")
        
        do {
            // 使用uv sync来确保所有依赖都已安装
            let syncOutput = try await executeCommand(
                arguments: ["sync"],
                workingDirectory: cliPath
            )
            
            logMessage("依赖同步完成")
            
            // 再次测试依赖
            let importTest = try await executeCommand(
                arguments: ["run", "python", "-c", "import typer; import mlx; print('Dependencies OK after sync')"],
                workingDirectory: cliPath
            )
            
            if importTest.contains("Dependencies OK after sync") {
                logMessage("✅ 依赖同步成功 - 所有模块现在都可用")
            } else {
                logMessage("❌ 依赖同步后仍有问题")
            }
        } catch {
            logMessage("❌ 依赖同步失败: \(error.localizedDescription)")
        }
    }
    
    func processSingleFile(_ fileURL: URL, config: TranslationConfig) async throws {
        logMessage("开始处理文件: \(fileURL.path)")
        
        let workingDirectory = fileURL.deletingLastPathComponent()
        
        var arguments = [
            "run", "python", "-m", "subtitle_translator.cli",
            "-i", fileURL.path,
            "-t", config.targetLanguage
        ]
        
        if !config.translationModel.isEmpty {
            arguments.append("--llm-model")
            arguments.append(config.translationModel)
        }
        
        if config.enableReflection {
            arguments.append("-r")
        }
        
        logMessage("执行处理命令，工作目录: \(workingDirectory.path)")
        
        let result = try await executeCommand(arguments: arguments, workingDirectory: workingDirectory.path)
        
        let outputAssFile = workingDirectory.appendingPathComponent(fileURL.deletingPathExtension().lastPathComponent + ".ass")
        let outputSrtFile = workingDirectory.appendingPathComponent(fileURL.deletingPathExtension().lastPathComponent + ".srt")
        
        if FileManager.default.fileExists(atPath: outputAssFile.path) {
            logMessage("输出ASS文件已生成: \(outputAssFile.path)")
        } else {
            logMessage("警告: ASS文件未生成: \(outputAssFile.path)")
        }
        
        if FileManager.default.fileExists(atPath: outputSrtFile.path) {
            logMessage("输出SRT文件已生成: \(outputSrtFile.path)")
        } else {
            logMessage("警告: SRT文件未生成: \(outputSrtFile.path)")
        }
    }
    
    private func executeCommand(arguments: [String], workingDirectory: String) async throws -> String {
        logMessage("=== 执行命令 ===")
        logMessage("参数: \(arguments)")
        logMessage("工作目录: \(workingDirectory)")
        
        // 同时记录到app.log
        appLog("执行CLI命令: \(arguments.joined(separator: " "))")
        appLog("工作目录: \(workingDirectory)")
        
        let process = Process()
        let pipe = Pipe()
        let errorPipe = Pipe()
        
        let uvPath = findUVPath()
        
        guard FileManager.default.fileExists(atPath: uvPath) else {
            let errorMsg = "uv可执行文件未找到: \(uvPath)"
            logMessage("错误: \(errorMsg)")
            appLog(errorMsg, level: .error)
            throw PythonBridgeError.launchFailed(errorMsg)
        }
        
        process.executableURL = URL(fileURLWithPath: uvPath)
        process.arguments = arguments
        process.currentDirectoryPath = workingDirectory
        process.standardOutput = pipe
        process.standardError = errorPipe
        
        var environment = ProcessInfo.processInfo.environment
        
        // 设置Python路径和模块路径
        environment["PYTHONPATH"] = "\(cliPath)/src"
        environment["UV_PROJECT_ENVIRONMENT"] = "\(cliPath)/.venv"
        
        // 确保虚拟环境的Python和站点包在PATH中
        let venvBinPath = "\(cliPath)/.venv/bin"
        
        // 设置PATH，确保能找到uv和其他工具，以及虚拟环境的Python
        let systemPaths = [
            venvBinPath,  // 优先使用虚拟环境的Python
            "/opt/homebrew/bin",
            "/usr/local/bin", 
            "/usr/bin",
            "/bin",
            "\(NSHomeDirectory())/.cargo/bin"
        ]
        environment["PATH"] = systemPaths.joined(separator: ":")
        
        // 设置虚拟环境激活标志
        environment["VIRTUAL_ENV"] = "\(cliPath)/.venv"
        
        // 设置用户主目录和语言环境
        environment["HOME"] = NSHomeDirectory()
        environment["LC_ALL"] = "en_US.UTF-8"
        environment["LANG"] = "en_US.UTF-8"
        
        // 禁用Python字节码缓存以避免权限问题
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        
        // 强制Python输出为unbuffered模式
        environment["PYTHONUNBUFFERED"] = "1"
        
        // 设置UV_PYTHON来确保使用正确的Python解释器
        environment["UV_PYTHON"] = "\(cliPath)/.venv/bin/python"
        
        logMessage("设置环境变量:")
        logMessage("- PYTHONPATH: \(environment["PYTHONPATH"] ?? "未设置")")
        logMessage("- PATH: \(environment["PATH"] ?? "未设置")")
        logMessage("- UV_PROJECT_ENVIRONMENT: \(environment["UV_PROJECT_ENVIRONMENT"] ?? "未设置")")
        logMessage("- VIRTUAL_ENV: \(environment["VIRTUAL_ENV"] ?? "未设置")")
        logMessage("- UV_PYTHON: \(environment["UV_PYTHON"] ?? "未设置")")
        
        process.environment = environment
        
        // 实时读取输出
        let outputHandle = pipe.fileHandleForReading
        let errorHandle = errorPipe.fileHandleForReading
        
        // 创建任务来实时读取输出
        Task {
            await readStreamData(from: outputHandle, prefix: "[CLI Output]", isError: false)
        }
        
        Task {
            await readStreamData(from: errorHandle, prefix: "[CLI Error]", isError: true)
        }
        
        try process.run()
        process.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        let errorOutput = String(data: errorData, encoding: .utf8) ?? ""
        
        logMessage("命令执行状态:")
        logMessage("- 退出代码: \(process.terminationStatus)")
        logMessage("- 输出长度: \(output.count) 字符")
        logMessage("- 错误输出长度: \(errorOutput.count) 字符")
        
        appLog("CLI命令执行完成，退出代码: \(process.terminationStatus)")
        
        if !output.isEmpty {
            logMessage("标准输出:")
            logMessage(String(output.prefix(1000))) // 限制日志长度
        }
        
        if !errorOutput.isEmpty {
            logMessage("错误输出:")
            logMessage(String(errorOutput.prefix(1000))) // 限制日志长度
        }
        
        if process.terminationStatus != 0 {
            var detailedError = "CLI执行失败 (退出代码: \(process.terminationStatus))"
            
            // 分析常见错误类型
            if errorOutput.contains("ModuleNotFoundError") {
                detailedError += "\n⚠️ Python模块缺失 - 请检查CLI依赖是否正确安装"
            } else if errorOutput.contains("Permission denied") {
                detailedError += "\n⚠️ 权限错误 - 应用可能没有访问文件的权限"
            } else if errorOutput.contains("FileNotFoundError") {
                detailedError += "\n⚠️ 文件未找到 - 请检查文件路径是否正确"
            } else if errorOutput.contains("API") || errorOutput.contains("401") || errorOutput.contains("403") {
                detailedError += "\n⚠️ API错误 - 请检查配置文件中的API密钥和Base URL"
            } else if uvPath != uvExecutable {
                detailedError += "\n⚠️ uv路径: \(uvPath)"
            }
            
            detailedError += "\n\n错误详情:\n\(errorOutput)"
            
            logMessage("详细错误分析: \(detailedError)")
            appLog("CLI执行失败: \(detailedError)", level: .error)
            throw PythonBridgeError.executionFailed(detailedError)
        }
        
        return output
    }
    
    // 新增：实时读取流数据
    private func readStreamData(from fileHandle: FileHandle, prefix: String, isError: Bool) async {
        let chunkSize = 1024
        
        while true {
            let data = fileHandle.availableData
            if data.isEmpty {
                break
            }
            
            if let string = String(data: data, encoding: .utf8) {
                let lines = string.components(separatedBy: .newlines)
                for line in lines where !line.isEmpty {
                    if isError {
                        appLog("\(prefix) \(line)", level: .warning)
                    } else {
                        appLog("\(prefix) \(line)")
                    }
                }
            }
            
            // 小延迟避免CPU占用过高
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        }
    }
    
    func checkEnvironment() async -> EnvironmentStatus {
        logMessage("开始环境检查")
        var status = EnvironmentStatus()
        
        let uvPath = findUVPath()
        status.pythonAvailable = FileManager.default.fileExists(atPath: uvPath)
        logMessage("uv可用性: \(status.pythonAvailable) (路径: \(uvPath))")
        
        status.cliAvailable = FileManager.default.fileExists(atPath: cliPath)
        logMessage("CLI可用性: \(status.cliAvailable) (路径: \(cliPath))")
        
        if status.cliAvailable {
            let pyprojectPath = "\(cliPath)/pyproject.toml"
            let pyprojectExists = FileManager.default.fileExists(atPath: pyprojectPath)
            logMessage("pyproject.toml存在: \(pyprojectExists)")
            
            let srcPath = "\(cliPath)/src/subtitle_translator"
            let srcExists = FileManager.default.fileExists(atPath: srcPath)
            logMessage("源代码目录存在: \(srcExists)")
        }
        
        // 检查配置文件是否存在和有效
        let configService = ConfigurationService()
        let validationResult = await configService.validateConfiguration()
        status.configurationValid = validationResult.isValid
        logMessage("配置验证结果: \(validationResult.isValid)")
        if !validationResult.isValid {
            logMessage("配置问题: \(validationResult.issues.joined(separator: ", "))")
        }
        
        logMessage("环境检查完成 - 就绪状态: \(status.isReady)")
        return status
    }
    
    private func findUVPath() -> String {
        let possiblePaths = [
            uvExecutable,
            "/usr/local/bin/uv",
            "/opt/homebrew/bin/uv",
            "/usr/bin/uv",
            "\(NSHomeDirectory())/.cargo/bin/uv"
        ]
        
        return possiblePaths.first { FileManager.default.fileExists(atPath: $0) } ?? uvExecutable
    }
    
    private func logMessage(_ message: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let logEntry = "[\(timestamp)] \(message)\n"
        
        print(logEntry.trimmingCharacters(in: .newlines))
        logger.info("\(message)")
        
        if let data = logEntry.data(using: .utf8) {
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
    }
    
    func getLogFilePath() -> String {
        return logFileURL.path
    }
    
    // MARK: - 公开的诊断方法
    
    func runDiagnostics() async -> [String] {
        var diagnostics: [String] = []
        
        logMessage("=== 开始诊断检查 ===")
        
        // 1. 检查基础路径
        diagnostics.append("CLI路径: \(cliPath)")
        diagnostics.append("CLI目录存在: \(FileManager.default.fileExists(atPath: cliPath))")
        
        let uvPath = findUVPath()
        diagnostics.append("uv路径: \(uvPath)")
        diagnostics.append("uv可执行: \(FileManager.default.fileExists(atPath: uvPath))")
        
        // 2. 检查Python环境
        let pyprojectPath = "\(cliPath)/pyproject.toml"
        diagnostics.append("pyproject.toml存在: \(FileManager.default.fileExists(atPath: pyprojectPath))")
        
        let srcPath = "\(cliPath)/src/subtitle_translator"
        diagnostics.append("Python包存在: \(FileManager.default.fileExists(atPath: srcPath))")
        
        // 3. 测试CLI基础命令
        do {
            let helpOutput = try await executeCommand(
                arguments: ["run", "python", "-m", "subtitle_translator.cli", "--help"],
                workingDirectory: cliPath
            )
            
            if helpOutput.contains("usage:") || helpOutput.contains("translate") {
                diagnostics.append("✅ CLI命令测试: 成功")
            } else {
                diagnostics.append("⚠️ CLI命令测试: 输出异常")
            }
        } catch {
            diagnostics.append("❌ CLI命令测试: 失败 - \(error.localizedDescription)")
        }
        
        // 4. 检查配置
        let configService = ConfigurationService()
        let configResult = await configService.validateConfiguration()
        diagnostics.append("配置有效性: \(configResult.isValid)")
        if !configResult.isValid {
            diagnostics.append("配置问题: \(configResult.issues.joined(separator: ", "))")
        }
        
        logMessage("=== 诊断检查完成 ===")
        return diagnostics
    }
}

enum PythonBridgeError: LocalizedError {
    case launchFailed(String)
    case executionFailed(String)
    case configurationError(String)
    
    var errorDescription: String? {
        switch self {
        case .launchFailed(let message):
            return "启动Python进程失败: \(message)"
        case .executionFailed(let message):
            return "执行失败: \(message)"
        case .configurationError(let message):
            return "配置错误: \(message)"
        }
    }
    
    var recoverySuggestion: String? {
        switch self {
        case .launchFailed:
            return "请检查uv是否正确安装，可以运行 'brew install uv' 进行安装"
        case .executionFailed:
            return "请检查文件权限和CLI配置，可以在终端中直接测试CLI命令"
        case .configurationError:
            return "请运行配置初始化或检查API密钥设置"
        }
    }
}

struct EnvironmentStatus {
    var pythonAvailable = false
    var cliAvailable = false
    var configurationValid = false
    
    var isReady: Bool {
        pythonAvailable && cliAvailable && configurationValid
    }
}