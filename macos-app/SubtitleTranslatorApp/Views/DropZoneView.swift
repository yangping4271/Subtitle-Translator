import SwiftUI
import UniformTypeIdentifiers

struct DropZoneView: View {
    @Binding var isDragOver: Bool
    let processingManager: ProcessingManager
    
    private let supportedFileTypes: [UTType] = {
        var types: [UTType] = [
            .mp3, .wav, .audio, // 基本音频格式
            .mpeg4Movie, .quickTimeMovie, .avi,  // 视频格式
            .movie, .video, // 通用视频类型
            .audiovisualContent // 通用音视频内容
        ]
        
        // 添加自定义格式
        if let srt = UTType(filenameExtension: "srt") { types.append(srt) }
        if let m4a = UTType(filenameExtension: "m4a") { types.append(m4a) }
        if let aac = UTType(filenameExtension: "aac") { types.append(aac) }
        if let flac = UTType(filenameExtension: "flac") { types.append(flac) }
        if let mkv = UTType(filenameExtension: "mkv") { types.append(mkv) }
        if let webm = UTType(filenameExtension: "webm") { types.append(webm) }
        
        return types
    }()
    
    var body: some View {
        RoundedRectangle(cornerRadius: 12)
            .stroke(
                isDragOver ? Color.blue : Color.gray.opacity(0.5),
                style: StrokeStyle(lineWidth: 2, dash: [10, 5])
            )
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(isDragOver ? Color.blue.opacity(0.1) : Color.gray.opacity(0.05))
            )
            .overlay(
                VStack(spacing: 16) {
                    Image(systemName: isDragOver ? "plus.circle.fill" : "plus.circle")
                        .font(.system(size: 48))
                        .foregroundColor(isDragOver ? .blue : .gray)
                    
                    VStack(spacing: 4) {
                        Text(isDragOver ? "释放文件开始处理" : "拖拽文件到此处")
                            .font(.headline)
                            .foregroundColor(isDragOver ? .blue : .primary)
                        
                        Text("支持 MP4, MP3, WAV, SRT 等格式")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    if !isDragOver {
                        Button("选择文件") {
                            selectFiles()
                        }
                        .buttonStyle(.bordered)
                    }
                }
            )
            .onDrop(of: supportedFileTypes, isTargeted: $isDragOver) { providers in
                appLog("开始处理拖拽，提供者数量: \(providers.count)")
                let result = handleDrop(providers: providers)
                appLog("拖拽处理完成，结果: \(result)")
                return result
            }
            .animation(.easeInOut(duration: 0.2), value: isDragOver)
    }
    
    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        appLog("handleDrop调用，providers数量: \(providers.count)")
        
        for (index, provider) in providers.enumerated() {
            appLog("处理第\(index + 1)个provider")
            
            // 打印支持的类型标识符
            let registeredTypes = provider.registeredTypeIdentifiers
            appLog("Provider支持的类型: \(registeredTypes)")
            
            // 检查是否有任何已注册的类型
            if let firstType = registeredTypes.first {
                appLog("尝试加载第一个类型: \(firstType)")
                
                provider.loadItem(forTypeIdentifier: firstType, options: nil) { (item, error) in
                    if let error = error {
                        appLog("加载错误: \(error)", level: .error)
                        return
                    }
                    
                    appLog("加载的item类型: \(type(of: item))")
                    
                    // 尝试各种方式获取URL
                    if let url = item as? URL {
                        DispatchQueue.main.async {
                            appLog("直接获取URL: \(url.path)")
                            self.processingManager.addFile(url)
                        }
                    } else if let data = item as? Data {
                        // 尝试从Data中获取URL
                        if let url = URL(dataRepresentation: data, relativeTo: nil) {
                            DispatchQueue.main.async {
                                appLog("从Data获取URL: \(url.path)")
                                self.processingManager.addFile(url)
                            }
                        } else {
                            appLog("Data无法转换为URL", level: .warning)
                        }
                    } else if let dict = item as? [String: Any] {
                        appLog("获取到字典数据: \(dict)")
                        // 检查字典中是否有文件路径信息
                        if let path = dict["path"] as? String {
                            let url = URL(fileURLWithPath: path)
                            DispatchQueue.main.async {
                                appLog("从字典获取文件路径: \(url.path)")
                                self.processingManager.addFile(url)
                            }
                        }
                    } else if let string = item as? String {
                        // 尝试将字符串作为路径或URL
                        if string.hasPrefix("file://") {
                            if let url = URL(string: string) {
                                DispatchQueue.main.async {
                                    appLog("从file:// URL字符串获取: \(url.path)")
                                    self.processingManager.addFile(url)
                                }
                            }
                        } else {
                            let url = URL(fileURLWithPath: string)
                            DispatchQueue.main.async {
                                appLog("从路径字符串获取: \(url.path)")
                                self.processingManager.addFile(url)
                            }
                        }
                    } else {
                        appLog("无法识别的item类型: \(String(describing: item))", level: .warning)
                    }
                }
            }
            
            // 同时尝试标准的文件URL加载方式
            if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                appLog("Provider也支持fileURL类型，尝试加载")
                provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { (item, error) in
                    if let error = error {
                        appLog("fileURL加载错误: \(error)", level: .error)
                        return
                    }
                    
                    if let url = item as? URL {
                        DispatchQueue.main.async {
                            appLog("通过fileURL获取: \(url.path)")
                            self.processingManager.addFile(url)
                        }
                    }
                }
            }
        }
        
        return true
    }
    
    private func selectFiles() {
        let openPanel = NSOpenPanel()
        openPanel.allowsMultipleSelection = true
        openPanel.canChooseDirectories = false
        openPanel.canChooseFiles = true
        openPanel.allowedContentTypes = supportedFileTypes
        
        if openPanel.runModal() == .OK {
            for url in openPanel.urls {
                processingManager.addFile(url)
            }
        }
    }
}

struct DropZoneView_Previews: PreviewProvider {
    static var previews: some View {
        DropZoneView(
            isDragOver: .constant(false),
            processingManager: ProcessingManager()
        )
        .frame(height: 200)
        .padding()
    }
}