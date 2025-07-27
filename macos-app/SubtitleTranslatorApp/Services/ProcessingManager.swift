import Foundation
import Combine

@MainActor
class ProcessingManager: ObservableObject {
    @Published var processingQueue: [ProcessingTask] = []
    @Published var completedTasks: [ProcessingTask] = []
    @Published var isProcessing = false
    
    private let pythonBridge = PythonBridgeService()
    private var processingCancellables = Set<AnyCancellable>()
    
    // 翻译配置
    @Published var config = TranslationConfig()
    
    init() {
        setupProcessingObserver()
    }
    
    // MARK: - 文件管理
    func addFile(_ url: URL) {
        appLog("ProcessingManager.addFile: \(url.path)")
        
        // 检查文件是否已存在
        let exists = processingQueue.contains { $0.url == url } || 
                    completedTasks.contains { $0.url == url }
        
        if !exists {
            let task = ProcessingTask(url: url)
            processingQueue.append(task)
            appLog("文件已添加到队列: \(url.lastPathComponent)")
            objectWillChange.send() // 通知UI更新
        } else {
            appLog("文件已存在，跳过: \(url.lastPathComponent)", level: .warning)
        }
    }
    
    func removeTask(_ task: ProcessingTask) {
        processingQueue.removeAll { $0.id == task.id }
    }
    
    func clearQueue() {
        processingQueue.removeAll()
    }
    
    func clearCompleted() {
        completedTasks.removeAll()
    }
    
    // MARK: - 处理控制
    func startProcessing() {
        guard !isProcessing && !processingQueue.isEmpty else { return }
        
        isProcessing = true
        processNextTask()
    }
    
    func stopProcessing() {
        isProcessing = false
        // 取消当前处理任务
        processingCancellables.removeAll()
    }
    
    private func processNextTask() {
        guard isProcessing, let nextTask = processingQueue.first else {
            isProcessing = false
            objectWillChange.send() // 通知UI更新
            return
        }
        
        // 开始处理任务
        Task {
            await processTask(nextTask)
            
            // 在主线程中更新状态
            await MainActor.run {
                // 移动到已完成列表
                if let index = processingQueue.firstIndex(where: { $0.id == nextTask.id }) {
                    processingQueue.remove(at: index)
                }
                completedTasks.append(nextTask)
                
                // 通知UI更新统计信息
                objectWillChange.send()
                
                // 继续处理下一个任务
                if !processingQueue.isEmpty {
                    processNextTask()
                } else {
                    isProcessing = false
                }
            }
        }
    }
    
    private func processTask(_ task: ProcessingTask) async {
        do {
            print("=== 开始处理任务 ===")
            print("文件: \(task.filename)")
            print("路径: \(task.url.path)")
            
            // 步骤1: 准备处理 (在主线程更新UI)
            await MainActor.run {
                task.updateProgress(step: 1, stage: "准备处理...")
            }
            
            // 步骤2-4: 使用PythonBridge处理单个文件
            await MainActor.run {
                task.updateProgress(step: 2, stage: "调用CLI处理...")
            }
            
            try await pythonBridge.processSingleFile(task.url, config: config)
            
            // 检查输出文件是否生成
            await MainActor.run {
                task.updateProgress(step: 3, stage: "验证输出文件...")
            }
            
            let outputAssExists = FileManager.default.fileExists(atPath: task.outputURL.path)
            let outputSrtPath = task.url.deletingPathExtension().appendingPathExtension("srt").path
            let outputSrtExists = FileManager.default.fileExists(atPath: outputSrtPath)
            
            if outputAssExists || outputSrtExists {
                // 完成
                await MainActor.run {
                    task.updateProgress(step: 4, stage: "处理完成")
                    task.markCompleted()
                }
                print("=== 任务处理完成 ===")
                print("输出文件: ASS=\(outputAssExists), SRT=\(outputSrtExists)")
            } else {
                throw ProcessingError.outputFileNotFound("未生成输出文件：\(task.outputURL.path)")
            }
            
        } catch {
            print("=== 任务处理失败 ===")
            print("错误: \(error)")
            await MainActor.run {
                task.markFailed(error: error.localizedDescription)
            }
        }
    }
    
    private func setupProcessingObserver() {
        // 监听配置变化等
    }
}

// MARK: - 统计信息
extension ProcessingManager {
    var queueCount: Int { processingQueue.count }
    var completedCount: Int { 
        completedTasks.filter { 
            if case .completed = $0.status { return true }
            return false 
        }.count 
    }
    var failedCount: Int { 
        completedTasks.filter { 
            if case .failed = $0.status { return true }
            return false 
        }.count 
    }
    
    // 添加成功率计算
    var successRate: Double {
        let total = completedTasks.count
        guard total > 0 else { return 0.0 }
        return Double(completedCount) / Double(total)
    }
    
    // 添加处理中的任务数
    var processingCount: Int {
        processingQueue.filter {
            if case .processing = $0.status { return true }
            return false
        }.count
    }
}

// MARK: - 环境检查
extension ProcessingManager {
    func checkEnvironment() async -> EnvironmentStatus {
        return await pythonBridge.checkEnvironment()
    }
    
    func getLogFilePath() -> String {
        return pythonBridge.getLogFilePath()
    }
    
    func runDiagnostics() async -> [String] {
        return await pythonBridge.runDiagnostics()
    }
}

// MARK: - 错误类型
enum ProcessingError: LocalizedError {
    case outputFileNotFound(String)
    case configurationInvalid(String)
    
    var errorDescription: String? {
        switch self {
        case .outputFileNotFound(let message):
            return "输出文件未生成: \(message)"
        case .configurationInvalid(let message):
            return "配置无效: \(message)"
        }
    }
}