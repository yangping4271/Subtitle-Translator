import SwiftUI

struct SidebarView: View {
    @ObservedObject var processingManager: ProcessingManager
    @StateObject private var configService = ConfigurationService()
    @State private var globalConfig = SubtitleTranslatorConfig()
    @State private var selectedLanguage = "zh"
    @State private var enableReflection = false
    @State private var selectedTranslationModel = ""
    @State private var selectedSplitModel = ""
    @State private var selectedSummaryModel = ""
    @State private var availableModels: [String] = []
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // 标题
            Text("字幕翻译器")
                .font(.title2)
                .fontWeight(.bold)
                .padding(.bottom, 10)
            
            // 配置状态指示
            if configService.isLoading {
                HStack {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("加载配置中...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            } else if let error = configService.lastError {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle")
                            .foregroundColor(.orange)
                        Text("配置加载警告")
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                    Text(error)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.orange.opacity(0.1))
                .cornerRadius(6)
            } else if configService.configurationExists() {
                HStack {
                    Image(systemName: "checkmark.circle")
                        .foregroundColor(.green)
                    Text("配置已加载")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            // 翻译设置
            GroupBox("翻译设置") {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("目标语言")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Picker("", selection: $selectedLanguage) {
                                ForEach(TranslationConfig.supportedLanguages, id: \.0) { languageCode, displayName in
                                    Text(displayName).tag(languageCode)
                                }
                            }
                            .pickerStyle(.menu)
                        }
                        
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("翻译模型")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                
                                Spacer()
                                
                                Button("刷新") {
                                    Task {
                                        await loadConfiguration()
                                    }
                                }
                                .buttonStyle(.plain)
                                .font(.caption2)
                                .foregroundColor(.blue)
                            }
                            
                            Picker("", selection: $selectedTranslationModel) {
                                ForEach(availableModels, id: \.self) { model in
                                    Text(model).tag(model)
                                }
                            }
                            .pickerStyle(.menu)
                            .disabled(availableModels.isEmpty)
                        }
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("断句模型")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .help("负责将长句分割成适合字幕显示的短句")
                            
                            Picker("", selection: $selectedSplitModel) {
                                ForEach(availableModels, id: \.self) { model in
                                    Text(model).tag(model)
                                }
                            }
                            .pickerStyle(.menu)
                            .disabled(availableModels.isEmpty)
                        }
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("总结模型")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .help("负责分析字幕内容并生成摘要")
                            
                            Picker("", selection: $selectedSummaryModel) {
                                ForEach(availableModels, id: \.self) { model in
                                    Text(model).tag(model)
                                }
                            }
                            .pickerStyle(.menu)
                            .disabled(availableModels.isEmpty)
                        }
                        
                        Toggle("启用反思翻译", isOn: $enableReflection)
                            .help("提高翻译质量，但会增加处理时间")
                        
                        // 配置信息显示
                        if !globalConfig.isValid {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Image(systemName: "gear")
                                        .foregroundColor(.blue)
                                    Text("需要配置")
                                        .font(.caption)
                                        .foregroundColor(.blue)
                                }
                                
                                Button("运行配置初始化") {
                                    // 这里可以集成配置初始化功能
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                        }
                    }
                    .padding(.vertical, 8)
                }
                .frame(maxHeight: 400) // 限制最大高度，启用滚动
            }
            
            // 处理统计
            GroupBox("处理统计") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("队列中:")
                        Spacer()
                        Text("\(processingManager.queueCount)")
                            .foregroundColor(.blue)
                    }
                    
                    HStack {
                        Text("已完成:")
                        Spacer()
                        Text("\(processingManager.completedCount)")
                            .foregroundColor(.green)
                    }
                    
                    HStack {
                        Text("失败:")
                        Spacer()
                        Text("\(processingManager.failedCount)")
                            .foregroundColor(.red)
                    }
                }
                .font(.caption)
                .padding(.vertical, 8)
            }
            
            Spacer()
            
            // 底部操作
            VStack(spacing: 8) {
                Button("清空队列") {
                    processingManager.clearQueue()
                    processingManager.clearCompleted()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                
                Button("打开输出文件夹") {
                    // 获取当前工作目录或最近处理的文件目录
                    let folderToOpen: URL
                    
                    if let lastProcessed = processingManager.completedTasks.last {
                        // 打开最后处理文件所在的目录
                        folderToOpen = lastProcessed.url.deletingLastPathComponent()
                    } else {
                        // 打开用户主目录
                        folderToOpen = FileManager.default.homeDirectoryForCurrentUser
                    }
                    
                    NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: folderToOpen.path)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                
                Button("打开配置文件夹") {
                    let configDirectory = URL(fileURLWithPath: configService.getConfigurationPath()).deletingLastPathComponent()
                    NSWorkspace.shared.open(configDirectory)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .onAppear {
            Task {
                await loadConfiguration()
            }
        }
    }
    
    // MARK: - 私有方法
    
    private func loadConfiguration() async {
        let config = await configService.loadConfiguration()
        
        await MainActor.run {
            globalConfig = config
            selectedTranslationModel = config.primaryTranslationModel
            selectedSplitModel = config.splitModel
            selectedSummaryModel = config.summaryModel
            availableModels = config.availableTranslationModels
            
            // 调试输出
            print("=== 配置加载调试信息 ===")
            print("翻译模型: \(config.primaryTranslationModel)")
            print("断句模型: \(config.splitModel)")
            print("总结模型: \(config.summaryModel)")
            print("可用模型列表: \(availableModels)")
            print("========================")
            
            // 如果当前选择的翻译模型不在可用列表中，选择第一个可用的
            if !availableModels.contains(selectedTranslationModel) && !availableModels.isEmpty {
                selectedTranslationModel = availableModels[0]
                print("翻译模型不在列表中，切换到: \(selectedTranslationModel)")
            }
            
            // 如果当前选择的断句模型不在可用列表中，选择第一个可用的
            if !availableModels.contains(selectedSplitModel) && !availableModels.isEmpty {
                selectedSplitModel = availableModels[0]
                print("断句模型不在列表中，切换到: \(selectedSplitModel)")
            }
            
            // 如果当前选择的总结模型不在可用列表中，选择第一个可用的
            if !availableModels.contains(selectedSummaryModel) && !availableModels.isEmpty {
                selectedSummaryModel = availableModels[0]
                print("总结模型不在列表中，切换到: \(selectedSummaryModel)")
            }
        }
    }
    
    // 获取当前翻译配置
    func getCurrentTranslationConfig() -> TranslationConfig {
        var config = TranslationConfig(from: globalConfig)
        config.targetLanguage = selectedLanguage
        config.translationModel = selectedTranslationModel
        config.splitModel = selectedSplitModel
        config.summaryModel = selectedSummaryModel
        config.enableReflection = enableReflection
        return config
    }
}

struct SidebarView_Previews: PreviewProvider {
    static var previews: some View {
        SidebarView(processingManager: ProcessingManager())
            .frame(width: 300, height: 600)
    }
}