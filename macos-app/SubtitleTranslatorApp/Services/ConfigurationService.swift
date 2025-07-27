import Foundation
import OSLog

class ConfigurationService: ObservableObject {
    @Published var isLoading = false
    @Published var lastError: String?
    @Published var validationResult: ConfigValidationResult?
    
    private let configPath: URL
    private let logger = Logger(subsystem: "com.subtitletranslator.app", category: "ConfigurationService")
    
    init() {
        // 全局配置文件路径
        let homeDirectory = FileManager.default.homeDirectoryForCurrentUser
        self.configPath = homeDirectory
            .appendingPathComponent(".config")
            .appendingPathComponent("subtitle_translator")
            .appendingPathComponent(".env")
        
        logger.info("配置服务初始化，配置文件路径: \\(self.configPath.path)")
    }
    
    // MARK: - 配置读取
    
    func loadConfiguration() async -> SubtitleTranslatorConfig {
        logger.info("开始加载配置")
        
        await MainActor.run {
            isLoading = true
            lastError = nil
        }
        
        defer {
            Task { @MainActor in
                isLoading = false
            }
        }
        
        do {
            let configData = try String(contentsOf: configPath, encoding: .utf8)
            let config = parseConfiguration(from: configData)
            
            logger.info("配置加载成功")
            await MainActor.run {
                lastError = nil
            }
            
            return config
        } catch {
            let errorMessage = "读取配置文件失败: \\(error.localizedDescription)"
            logger.error("\\(errorMessage)")
            
            await MainActor.run {
                lastError = errorMessage
            }
            
            // 返回默认配置
            return SubtitleTranslatorConfig()
        }
    }
    
    func saveConfiguration(_ config: SubtitleTranslatorConfig) async throws {
        logger.info("开始保存配置")
        
        let configContent = generateConfigContent(from: config)
        
        // 确保目录存在
        let configDirectory = configPath.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: configDirectory, withIntermediateDirectories: true)
        
        try configContent.write(to: configPath, atomically: true, encoding: .utf8)
        
        logger.info("配置保存成功")
    }
    
    // MARK: - 配置验证
    
    func validateConfiguration() async -> ConfigValidationResult {
        logger.info("开始配置验证")
        
        let config = await loadConfiguration()
        var result = ConfigValidationResult()
        
        // 检查配置文件存在
        result.configFileExists = configurationExists()
        logger.debug("配置文件存在: \\(result.configFileExists)")
        
        // 检查 API 配置
        result.hasValidAPIKey = !config.openaiAPIKey.isEmpty
        result.hasValidBaseURL = !config.openaiBaseURL.isEmpty && isValidURL(config.openaiBaseURL)
        logger.debug("API 密钥有效: \\(result.hasValidAPIKey), Base URL 有效: \\(result.hasValidBaseURL)")
        
        // 检查模型配置
        result.hasValidModels = !config.primaryTranslationModel.isEmpty
        logger.debug("模型配置有效: \\(result.hasValidModels)")
        
        // 收集验证问题
        var issues: [String] = []
        
        if !result.configFileExists {
            issues.append("配置文件不存在")
        }
        
        if !result.hasValidAPIKey {
            issues.append("API 密钥未设置")
        }
        
        if !result.hasValidBaseURL {
            issues.append("API Base URL 无效")
        }
        
        if !result.hasValidModels {
            issues.append("模型配置缺失")
        }
        
        result.issues = issues
        result.isValid = issues.isEmpty
        
        await MainActor.run {
            self.validationResult = result
        }
        
        logger.info("配置验证完成 - 有效: \\(result.isValid), 问题数: \\(issues.count)")
        return result
    }
    
    private func isValidURL(_ urlString: String) -> Bool {
        guard let url = URL(string: urlString) else { return false }
        return url.scheme != nil && url.host != nil
    }
    
    // MARK: - 私有方法
    
    private func parseConfiguration(from content: String) -> SubtitleTranslatorConfig {
        var config = SubtitleTranslatorConfig()
        
        let lines = content.components(separatedBy: .newlines)
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            
            // 跳过注释和空行
            if trimmed.isEmpty || trimmed.hasPrefix("#") {
                continue
            }
            
            // 解析 KEY=VALUE 格式
            let components = trimmed.components(separatedBy: "=")
            guard components.count >= 2 else { continue }
            
            let key = components[0].trimmingCharacters(in: .whitespaces)
            let value = components.dropFirst().joined(separator: "=").trimmingCharacters(in: .whitespaces)
            
            switch key {
            case "OPENAI_BASE_URL":
                config.openaiBaseURL = value
            case "OPENAI_API_KEY":
                config.openaiAPIKey = value
            case "HF_ENDPOINT":
                config.hfEndpoint = value
            case "SPLIT_MODEL":
                config.splitModel = value
            case "TRANSLATION_MODEL":
                config.translationModel = value
            case "SUMMARY_MODEL":
                config.summaryModel = value
            case "LLM_MODEL":
                config.llmModel = value
            default:
                break
            }
        }
        
        return config
    }
    
    private func generateConfigContent(from config: SubtitleTranslatorConfig) -> String {
        return """
        # Subtitle Translator 配置文件
        # 由 translate init 命令自动生成
        
        # ======== API 配置 ========
        # API 基础URL
        OPENAI_BASE_URL=\\(config.openaiBaseURL)
        
        # API 密钥
        OPENAI_API_KEY=\\(config.openaiAPIKey)
        
        # Hugging Face 镜像站地址 (用于模型下载)
        # 留空使用默认官方地址，设置后可提高国内下载成功率
        HF_ENDPOINT=\\(config.hfEndpoint)
        
        # ======== 模型配置 ========
        # 断句模型 - 负责将长句分割成适合字幕显示的短句
        SPLIT_MODEL=\\(config.splitModel)
        
        # 翻译模型 - 负责将字幕翻译成目标语言
        TRANSLATION_MODEL=\\(config.translationModel)
        
        # 总结模型 - 负责分析字幕内容并生成摘要
        SUMMARY_MODEL=\\(config.summaryModel)
        
        # 兼容性：默认模型 (如果上述模型未设置，将使用此模型)
        LLM_MODEL=\\(config.llmModel)
        
        # ======== 使用说明 ========
        # 1. 你现在可以在任意目录下运行 translate 命令
        # 2. 如需修改配置，可以编辑此文件或重新运行 translate init
        # 3. 分别配置的模型会优先使用，如未设置则回退到 LLM_MODEL
        # 4. HF_ENDPOINT 用于设置 Hugging Face 镜像站，可提高模型下载成功率
        """
    }
    
    // MARK: - 便利方法
    
    func configurationExists() -> Bool {
        let exists = FileManager.default.fileExists(atPath: configPath.path)
        logger.debug("检查配置文件存在: \\(exists)")
        return exists
    }
    
    func getConfigurationPath() -> String {
        return configPath.path
    }
    
    // 新增: 获取配置目录
    func getConfigurationDirectory() -> String {
        return configPath.deletingLastPathComponent().path
    }
    
    // 新增: 创建配置模板
    func createConfigurationTemplate() async throws {
        logger.info("创建配置模板")
        
        let templateConfig = SubtitleTranslatorConfig(
            openaiBaseURL: "https://api.openai.com/v1",
            openaiAPIKey: "your-api-key-here"
        )
        
        try await saveConfiguration(templateConfig)
    }
}

// MARK: - 配置数据结构

struct SubtitleTranslatorConfig {
    var openaiBaseURL: String
    var openaiAPIKey: String
    var hfEndpoint: String
    var splitModel: String
    var translationModel: String
    var summaryModel: String
    var llmModel: String
    
    init(
        openaiBaseURL: String = "https://api.openai.com/v1",
        openaiAPIKey: String = "",
        hfEndpoint: String = "https://huggingface.co",
        splitModel: String = "gpt-4o-mini",
        translationModel: String = "gpt-4o",
        summaryModel: String = "gpt-4o-mini",
        llmModel: String = "gpt-4o-mini"
    ) {
        self.openaiBaseURL = openaiBaseURL
        self.openaiAPIKey = openaiAPIKey
        self.hfEndpoint = hfEndpoint
        self.splitModel = splitModel
        self.translationModel = translationModel
        self.summaryModel = summaryModel
        self.llmModel = llmModel
    }
    
    // 获取可用的翻译模型列表
    var availableTranslationModels: [String] {
        // 从配置中提取不重复的模型
        var models = Set<String>()
        
        // 添加配置文件中的实际模型
        if !translationModel.isEmpty {
            models.insert(translationModel)
        }
        if !splitModel.isEmpty {
            models.insert(splitModel)
        }
        if !summaryModel.isEmpty {
            models.insert(summaryModel)
        }
        if !llmModel.isEmpty {
            models.insert(llmModel)
        }
        
        // 添加一些常见的模型选项（只有在不为空时才添加）
        let commonModels = [
            "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
            "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite",
            "openai/gpt-4.1-nano", "claude-3-haiku", "claude-3-sonnet"
        ]
        
        // 只添加非空的常见模型
        for model in commonModels {
            if !model.isEmpty {
                models.insert(model)
            }
        }
        
        return Array(models).sorted()
    }
    
    // 获取主要翻译模型
    var primaryTranslationModel: String {
        return !translationModel.isEmpty ? translationModel : llmModel
    }
    
    // 验证配置是否完整
    var isValid: Bool {
        return !openaiAPIKey.isEmpty && !primaryTranslationModel.isEmpty && isValidURL(openaiBaseURL)
    }
    
    private func isValidURL(_ urlString: String) -> Bool {
        guard let url = URL(string: urlString) else { return false }
        return url.scheme != nil && url.host != nil
    }
}

// MARK: - 配置验证结果
struct ConfigValidationResult {
    var isValid = false
    var configFileExists = false
    var hasValidAPIKey = false
    var hasValidBaseURL = false
    var hasValidModels = false
    var issues: [String] = []
    
    var summary: String {
        if isValid {
            return "配置有效"
        } else {
            return "配置有 \\(issues.count) 个问题"
        }
    }
}