import Foundation
import SwiftUI

// MARK: - 处理任务状态
enum ProcessingStatus {
    case pending
    case processing(stage: String)
    case completed
    case failed(error: String)
    
    var description: String {
        switch self {
        case .pending:
            return "等待处理"
        case .processing(let stage):
            return stage
        case .completed:
            return "处理完成"
        case .failed:
            return "处理失败"
        }
    }
    
    var color: Color {
        switch self {
        case .pending:
            return .orange
        case .processing:
            return .blue
        case .completed:
            return .green
        case .failed:
            return .red
        }
    }
}

// MARK: - 文件类型
enum FileType {
    case audio
    case video
    case subtitle
    
    var systemImageName: String {
        switch self {
        case .audio:
            return "waveform"
        case .video:
            return "video"
        case .subtitle:
            return "captions.bubble"
        }
    }
    
    var color: Color {
        switch self {
        case .audio:
            return .blue
        case .video:
            return .purple
        case .subtitle:
            return .green
        }
    }
    
    static func from(url: URL) -> FileType {
        let fileExtension = url.pathExtension.lowercased()
        
        switch fileExtension {
        case "mp3", "wav", "m4a", "aac", "flac":
            return .audio
        case "mp4", "mov", "mkv", "avi", "webm":
            return .video
        case "srt":
            return .subtitle
        default:
            return .video // 默认当作视频处理
        }
    }
}

// MARK: - 处理任务
class ProcessingTask: ObservableObject, Identifiable {
    let id = UUID()
    let url: URL
    let filename: String
    let fileType: FileType
    
    @Published var status: ProcessingStatus = .pending
    @Published var progress: Double = 0.0
    @Published var currentStep: Int = 0
    @Published var totalSteps: Int = 4 // 转录、分析、翻译、生成字幕
    @Published var outputFilePath: String = "" // 输出文件路径
    
    init(url: URL) {
        self.url = url
        self.filename = url.lastPathComponent
        self.fileType = FileType.from(url: url)
        // 预设输出文件路径（.ass字幕文件）
        self.outputFilePath = url.deletingPathExtension().appendingPathExtension("ass").path
    }
    
    func updateProgress(step: Int, stage: String) {
        // 确保在主线程中更新
        if Thread.isMainThread {
            self.currentStep = step
            self.progress = Double(step) / Double(self.totalSteps)
            self.status = .processing(stage: stage)
        } else {
            DispatchQueue.main.async {
                self.currentStep = step
                self.progress = Double(step) / Double(self.totalSteps)
                self.status = .processing(stage: stage)
            }
        }
    }
    
    func markCompleted() {
        if Thread.isMainThread {
            self.status = .completed
            self.progress = 1.0
            self.currentStep = self.totalSteps
        } else {
            DispatchQueue.main.async {
                self.status = .completed
                self.progress = 1.0
                self.currentStep = self.totalSteps
            }
        }
    }
    
    func markFailed(error: String) {
        if Thread.isMainThread {
            self.status = .failed(error: error)
        } else {
            DispatchQueue.main.async {
                self.status = .failed(error: error)
            }
        }
    }
    
    // 获取输出文件URL
    var outputURL: URL {
        return URL(fileURLWithPath: outputFilePath)
    }
}

// MARK: - 翻译配置
struct TranslationConfig {
    var targetLanguage: String = "zh"
    var translationModel: String = "gpt-4o"
    var splitModel: String = "gpt-4o-mini"
    var summaryModel: String = "gpt-4o-mini"
    var enableReflection: Bool = false
    var outputDirectory: URL?
    
    // 从全局配置初始化
    init(from globalConfig: SubtitleTranslatorConfig? = nil) {
        if let config = globalConfig {
            self.translationModel = config.primaryTranslationModel
            self.splitModel = config.splitModel
            self.summaryModel = config.summaryModel
        }
    }
    
    // 默认初始化器
    init() {
        // 使用默认值
    }
    
    // 语言映射
    static let languageMap: [String: String] = [
        "zh": "简体中文",
        "zh-tw": "繁体中文", 
        "ja": "日文",
        "ko": "韩文",
        "en": "英文",
        "fr": "法文",
        "de": "德文",
        "es": "西班牙文",
        "pt": "葡萄牙文",
        "ru": "俄文",
        "it": "意大利文",
        "ar": "阿拉伯文",
        "th": "泰文",
        "vi": "越南文"
    ]
    
    // 获取支持的语言列表
    static var supportedLanguages: [(String, String)] {
        return languageMap.map { ($0.key, $0.value) }.sorted { $0.0 < $1.0 }
    }
    
    // 获取语言显示名称
    func getLanguageDisplayName(_ languageCode: String) -> String {
        return Self.languageMap[languageCode] ?? languageCode
    }
}