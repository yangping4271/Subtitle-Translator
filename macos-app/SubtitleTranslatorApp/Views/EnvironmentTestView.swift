import SwiftUI
import OSLog

struct EnvironmentTestView: View {
    @StateObject private var processingManager = ProcessingManager()
    @State private var environmentStatus: EnvironmentStatus?
    @State private var diagnosticResults: [String] = []
    @State private var isLoading = false
    @State private var showDetails = false
    @State private var showDiagnostics = false
    
    private let logger = Logger(subsystem: "com.subtitletranslator.app", category: "EnvironmentTest")
    
    var body: some View {
        VStack(spacing: 16) {
            Text("环境检查")
                .font(.title2)
                .fontWeight(.bold)
            
            if isLoading {
                ProgressView("检查中...")
                    .progressViewStyle(CircularProgressViewStyle())
                    .scaleEffect(1.2)
            } else if let status = environmentStatus {
                VStack(spacing: 12) {
                    // 总体状态
                    HStack {
                        Image(systemName: status.isReady ? "checkmark.circle.fill" : "xmark.circle.fill")
                            .foregroundColor(status.isReady ? .green : .red)
                            .font(.title)
                        
                        Text(status.isReady ? "环境就绪" : "环境需要配置")
                            .font(.headline)
                            .foregroundColor(status.isReady ? .green : .red)
                    }
                    
                    // 详细状态
                    VStack(alignment: .leading, spacing: 8) {
                        StatusRow(
                            title: "Python (uv)",
                            status: status.pythonAvailable,
                            description: status.pythonAvailable ? "uv 可执行文件已找到" : "未找到 uv 可执行文件"
                        )
                        
                        StatusRow(
                            title: "CLI 程序",
                            status: status.cliAvailable,
                            description: status.cliAvailable ? "字幕翻译 CLI 已就绪" : "CLI 程序不可用"
                        )
                        
                        StatusRow(
                            title: "配置文件",
                            status: status.configurationValid,
                            description: status.configurationValid ? "API 配置有效" : "配置缺失或无效"
                        )
                    }
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color(NSColor.controlBackgroundColor))
                    )
                    
                    // 操作按钮
                    HStack(spacing: 12) {
                        Button("重新检查") {
                            checkEnvironment()
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("详细诊断") {
                            runDetailedDiagnostics()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("查看日志") {
                            showDetails = true
                        }
                        .buttonStyle(.bordered)
                        
                        if !status.isReady {
                            Button("修复问题") {
                                showFixSuggestions()
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
            } else {
                Button("开始检查") {
                    checkEnvironment()
                }
                .buttonStyle(.borderedProminent)
                .font(.headline)
            }
        }
        .padding()
        .frame(maxWidth: 400)
        .onAppear {
            checkEnvironment()
        }
        .sheet(isPresented: $showDetails) {
            LogViewerSheet(logFilePath: processingManager.getLogFilePath())
        }
        .sheet(isPresented: $showDiagnostics) {
            DiagnosticsSheet(diagnosticResults: diagnosticResults)
        }
    }
    
    private func checkEnvironment() {
        isLoading = true
        logger.info("开始环境检查")
        
        Task {
            let status = await processingManager.checkEnvironment()
            
            await MainActor.run {
                self.environmentStatus = status
                self.isLoading = false
                logger.info("环境检查完成 - 就绪状态: \\(status.isReady)")
            }
        }
    }
    
    private func runDetailedDiagnostics() {
        isLoading = true
        logger.info("开始详细诊断")
        
        Task {
            let results = await processingManager.runDiagnostics()
            
            await MainActor.run {
                self.diagnosticResults = results
                self.isLoading = false
                self.showDiagnostics = true
                logger.info("详细诊断完成，共 \\(results.count) 项结果")
            }
        }
    }
    
    private func showFixSuggestions() {
        guard let status = environmentStatus else { return }
        
        var suggestions: [String] = []
        
        if !status.pythonAvailable {
            suggestions.append("安装 uv: brew install uv")
        }
        
        if !status.cliAvailable {
            suggestions.append("重新构建应用，确保 CLI 程序被正确打包")
        }
        
        if !status.configurationValid {
            suggestions.append("运行配置初始化或检查 API 密钥设置")
        }
        
        let alert = NSAlert()
        alert.messageText = "修复建议"
        alert.informativeText = suggestions.joined(separator: "\\n\\n")
        alert.addButton(withTitle: "确定")
        alert.runModal()
    }
}

struct StatusRow: View {
    let title: String
    let status: Bool
    let description: String
    
    var body: some View {
        HStack {
            Image(systemName: status ? "checkmark.circle.fill" : "xmark.circle.fill")
                .foregroundColor(status ? .green : .red)
                .font(.system(size: 16))
            
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
        }
    }
}

// 预览
struct EnvironmentTestView_Previews: PreviewProvider {
    static var previews: some View {
        EnvironmentTestView()
            .frame(width: 500, height: 400)
    }
}

struct DiagnosticsSheet: View {
    let diagnosticResults: [String]
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text("详细诊断结果")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button("关闭") {
                    presentationMode.wrappedValue.dismiss()
                }
            }
            
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(diagnosticResults.enumerated()), id: \.offset) { index, result in
                        HStack(alignment: .top) {
                            if result.contains("✅") {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                            } else if result.contains("⚠️") {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundColor(.orange)
                            } else if result.contains("❌") {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundColor(.red)
                            } else {
                                Image(systemName: "info.circle")
                                    .foregroundColor(.blue)
                            }
                            
                            Text(result)
                                .font(.system(.body, design: .monospaced))
                                .textSelection(.enabled)
                            
                            Spacer()
                        }
                        .padding(.vertical, 2)
                    }
                }
                .padding()
            }
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(NSColor.textBackgroundColor))
            )
            
            Button("复制全部") {
                let allResults = diagnosticResults.joined(separator: "\n")
                NSPasteboard.general.setString(allResults, forType: .string)
            }
            .buttonStyle(.bordered)
        }
        .padding()
        .frame(width: 600, height: 500)
    }
}

struct LogViewerSheet: View {
    let logFilePath: String
    @State private var logContent = ""
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text("日志查看器")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button("刷新") {
                    loadLogContent()
                }
                .buttonStyle(.bordered)
                
                Button("关闭") {
                    presentationMode.wrappedValue.dismiss()
                }
            }
            
            ScrollView {
                Text(logContent)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(NSColor.textBackgroundColor))
            )
            
            HStack {
                Button("复制日志") {
                    NSPasteboard.general.setString(logContent, forType: .string)
                }
                .buttonStyle(.bordered)
                
                Button("打开文件") {
                    NSWorkspace.shared.open(URL(fileURLWithPath: logFilePath))
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
        .frame(width: 700, height: 500)
        .onAppear {
            loadLogContent()
        }
    }
    
    private func loadLogContent() {
        do {
            logContent = try String(contentsOfFile: logFilePath, encoding: .utf8)
        } catch {
            logContent = "无法读取日志文件: \(error.localizedDescription)\n\n文件路径: \(logFilePath)"
        }
    }
}