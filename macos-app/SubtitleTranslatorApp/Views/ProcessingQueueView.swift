import SwiftUI

struct ProcessingQueueView: View {
    @ObservedObject var processingManager: ProcessingManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("处理队列")
                    .font(.headline)
                
                Spacer()
                
                if !processingManager.processingQueue.isEmpty {
                    Text("\\(processingManager.processingQueue.count) 个文件")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            if processingManager.processingQueue.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "tray")
                        .font(.system(size: 32))
                        .foregroundColor(.gray)
                    
                    Text("暂无待处理文件")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                    
                    Text("拖拽文件到上方区域开始处理")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 100)
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(processingManager.processingQueue) { task in
                            ProcessingTaskRow(task: task)
                        }
                    }
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(NSColor.controlBackgroundColor))
        )
    }
}

struct ProcessingTaskRow: View {
    @ObservedObject var task: ProcessingTask
    
    var body: some View {
        HStack(spacing: 12) {
            // 文件图标
            Image(systemName: task.fileType.systemImageName)
                .font(.system(size: 20))
                .foregroundColor(task.fileType.color)
                .frame(width: 24)
            
            // 文件信息
            VStack(alignment: .leading, spacing: 2) {
                Text(task.filename)
                    .font(.subheadline)
                    .lineLimit(1)
                
                Text(task.status.description)
                    .font(.caption)
                    .foregroundColor(task.status.color)
            }
            
            Spacer()
            
            // 进度或状态
            switch task.status {
            case .pending:
                Text("等待中")
                    .font(.caption)
                    .foregroundColor(.orange)
                
            case .processing(let stage):
                VStack(alignment: .trailing, spacing: 2) {
                    Text(stage)
                        .font(.caption)
                        .foregroundColor(.blue)
                    
                    ProgressView(value: task.progress)
                        .progressViewStyle(LinearProgressViewStyle())
                        .frame(width: 80)
                }
                
            case .completed:
                VStack(alignment: .trailing, spacing: 2) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    
                    Button("查看文件") {
                        // 查找所有可能的输出文件
                        let baseURL = task.url.deletingPathExtension()
                        let possibleFiles = [
                            baseURL.appendingPathExtension("ass"),
                            baseURL.appendingPathExtension("srt")
                        ]
                        
                        // 找到第一个存在的文件进行显示
                        if let existingFile = possibleFiles.first(where: { FileManager.default.fileExists(atPath: $0.path) }) {
                            NSWorkspace.shared.selectFile(existingFile.path, inFileViewerRootedAtPath: task.url.deletingLastPathComponent().path)
                        } else {
                            // 如果没有找到输出文件，就打开原文件所在目录
                            NSWorkspace.shared.selectFile(task.url.path, inFileViewerRootedAtPath: task.url.deletingLastPathComponent().path)
                        }
                    }
                    .font(.caption2)
                    .buttonStyle(.plain)
                    .foregroundColor(.blue)
                }
                
            case .failed(let error):
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.red)
                    .help(error)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(Color(NSColor.textBackgroundColor))
        )
    }
}

// 预览
struct ProcessingQueueView_Previews: PreviewProvider {
    static var previews: some View {
        let manager = ProcessingManager()
        return ProcessingQueueView(processingManager: manager)
            .frame(width: 400, height: 300)
            .padding()
    }
}