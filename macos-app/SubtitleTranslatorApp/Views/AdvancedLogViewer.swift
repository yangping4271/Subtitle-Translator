import SwiftUI
import OSLog

struct AdvancedLogViewer: View {
    @State private var logContent = ""
    @State private var filter = LogFilter()
    @State private var isLoading = false
    @State private var showExportSheet = false
    @State private var exportURL: URL?
    @State private var autoRefresh = false
    @State private var refreshTimer: Timer?
    
    @Environment(\\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // 工具栏
                HStack {
                    // 搜索框
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundColor(.secondary)
                        TextField("搜索日志...", text: $filter.searchText)
                            .textFieldStyle(.plain)
                    }
                    .padding(8)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color(NSColor.controlBackgroundColor))
                    )
                    .frame(maxWidth: 200)
                    
                    Spacer()
                    
                    // 自动刷新开关
                    HStack {
                        Image(systemName: autoRefresh ? "arrow.clockwise.circle.fill" : "arrow.clockwise.circle")
                            .foregroundColor(autoRefresh ? .blue : .secondary)
                        Text("自动刷新")
                            .font(.caption)
                    }
                    .onTapGesture {
                        toggleAutoRefresh()
                    }
                    
                    // 级别过滤器
                    Menu {
                        ForEach(LogLevel.allCases, id: \\.self) { level in
                            Button {
                                toggleLevel(level)
                            } label: {
                                HStack {
                                    if filter.levels.contains(level) {
                                        Image(systemName: "checkmark")
                                    }
                                    Text("\\(level.emoji) \\(level.rawValue.capitalized)")
                                }
                            }
                        }
                    } label: {
                        Image(systemName: "line.3.horizontal.decrease.circle")
                    }
                    .menuStyle(.borderlessButton)
                    
                    // 操作按钮
                    HStack(spacing: 8) {
                        Button("刷新") {
                            loadLogContent()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("导出") {
                            exportLogs()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("清空") {
                            clearLogs()
                        }
                        .buttonStyle(.bordered)
                        .foregroundColor(.red)
                    }
                }
                .padding()
                .background(Color(NSColor.controlBackgroundColor))
                
                Divider()
                
                // 日志内容
                if isLoading {
                    VStack {
                        ProgressView("加载日志...")
                        Spacer()
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ScrollView {
                        ScrollViewReader { proxy in
                            VStack(alignment: .leading, spacing: 0) {
                                ForEach(parseLogEntries(logContent), id: \\.timestamp) { entry in
                                    if filter.matches(entry) {
                                        LogEntryRow(entry: entry)
                                    }
                                }
                            }
                            .id("bottom")
                            .onAppear {
                                proxy.scrollTo("bottom", anchor: .bottom)
                            }
                            .onChange(of: logContent) { _ in
                                if autoRefresh {
                                    proxy.scrollTo("bottom", anchor: .bottom)
                                }
                            }
                        }
                    }
                    .background(Color(NSColor.textBackgroundColor))
                }
                
                // 状态栏
                HStack {
                    let visibleEntries = parseLogEntries(logContent).filter(filter.matches)
                    Text("显示 \\(visibleEntries.count) 条日志")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text("日志文件: \\(LogManager.shared.getLogFileURL().lastPathComponent)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal)
                .padding(.vertical, 4)
                .background(Color(NSColor.controlBackgroundColor))
            }
            .navigationTitle("应用日志")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("关闭") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("在Finder中显示") {
                        NSWorkspace.shared.selectFile(
                            LogManager.shared.getLogFileURL().path,
                            inFileViewerRootedAtPath: LogManager.shared.getLogDirectory().path
                        )
                    }
                }
            }
        }
        .frame(minWidth: 800, minHeight: 600)
        .onAppear {
            loadLogContent()
        }
        .sheet(isPresented: $showExportSheet) {
            if let exportURL = exportURL {
                ExportSuccessSheet(fileURL: exportURL)
            }
        }
        .onDisappear {
            stopAutoRefresh()
        }
    }
    
    private func loadLogContent() {
        isLoading = true
        
        DispatchQueue.global(qos: .background).async {
            let content = LogManager.shared.getAllLogs()
            
            DispatchQueue.main.async {
                self.logContent = content
                self.isLoading = false
            }
        }
    }
    
    private func parseLogEntries(_ content: String) -> [LogEntry] {
        let lines = content.components(separatedBy: .newlines)
        var entries: [LogEntry] = []
        
        let dateFormatter = ISO8601DateFormatter()
        
        for line in lines {
            // 解析格式: [timestamp] [LEVEL] [category] message
            let pattern = "^\\[([^\\]]+)\\] \\[([^\\]]+)\\] \\[([^\\]]+)\\] (.+)$"
            if let regex = try? NSRegularExpression(pattern: pattern),
               let match = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) {
                
                let timestampString = String(line[Range(match.range(at: 1), in: line)!])
                let levelString = String(line[Range(match.range(at: 2), in: line)!])
                let categoryString = String(line[Range(match.range(at: 3), in: line)!])
                let messageString = String(line[Range(match.range(at: 4), in: line)!])
                
                if let timestamp = dateFormatter.date(from: timestampString),
                   let level = LogLevel(rawValue: levelString.lowercased()) {
                    let entry = LogEntry(
                        timestamp: timestamp,
                        level: level,
                        category: categoryString,
                        message: messageString
                    )
                    entries.append(entry)
                }
            }
        }
        
        return entries.sorted { $0.timestamp < $1.timestamp }
    }
    
    private func toggleLevel(_ level: LogLevel) {
        if filter.levels.contains(level) {
            filter.levels.remove(level)
        } else {
            filter.levels.insert(level)
        }
    }
    
    private func toggleAutoRefresh() {
        autoRefresh.toggle()
        
        if autoRefresh {
            refreshTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
                loadLogContent()
            }
        } else {
            stopAutoRefresh()
        }
    }
    
    private func stopAutoRefresh() {
        refreshTimer?.invalidate()
        refreshTimer = nil
        autoRefresh = false
    }
    
    private func exportLogs() {
        if let exportURL = LogManager.shared.exportLogs() {
            self.exportURL = exportURL
            showExportSheet = true
        }
    }
    
    private func clearLogs() {
        let alert = NSAlert()
        alert.messageText = "清空日志"
        alert.informativeText = "确定要清空所有日志吗？此操作不可撤销。"
        alert.addButton(withTitle: "清空")
        alert.addButton(withTitle: "取消")
        alert.alertStyle = .warning
        
        if alert.runModal() == .alertFirstButtonReturn {
            LogManager.shared.clearLogs()
            loadLogContent()
        }
    }
}

struct LogEntryRow: View {
    let entry: LogEntry
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                // 时间戳
                Text(entry.timestamp, style: .time)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.secondary)
                    .frame(width: 70, alignment: .leading)
                
                // 级别图标
                Text(entry.level.emoji)
                    .font(.caption)
                
                // 分类
                Text(entry.category)
                    .font(.caption)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(
                        RoundedRectangle(cornerRadius: 3)
                            .fill(Color.blue.opacity(0.1))
                    )
                    .foregroundColor(.blue)
                
                // 消息预览
                Text(entry.message)
                    .font(.system(.caption, design: .monospaced))
                    .lineLimit(isExpanded ? nil : 1)
                    .frame(maxWidth: .infinity, alignment: .leading)
                
                // 展开按钮
                if entry.message.contains("\\n") || entry.message.count > 100 {
                    Button {
                        isExpanded.toggle()
                    } label: {
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .font(.caption2)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            .background(
                entry.level == .error ? Color.red.opacity(0.1) :
                entry.level == .warning ? Color.orange.opacity(0.1) :
                Color.clear
            )
        }
        .contentShape(Rectangle())
        .onTapGesture {
            if entry.message.contains("\\n") || entry.message.count > 100 {
                isExpanded.toggle()
            }
        }
    }
}

struct ExportSuccessSheet: View {
    let fileURL: URL
    @Environment(\\.dismiss) private var dismiss
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 48))
                .foregroundColor(.green)
            
            Text("日志导出成功")
                .font(.headline)
            
            Text("文件已保存到:")
                .foregroundColor(.secondary)
            
            Text(fileURL.path)
                .font(.system(.caption, design: .monospaced))
                .padding(8)
                .background(Color(NSColor.controlBackgroundColor))
                .cornerRadius(4)
            
            HStack(spacing: 12) {
                Button("在Finder中显示") {
                    NSWorkspace.shared.selectFile(fileURL.path, inFileViewerRootedAtPath: fileURL.deletingLastPathComponent().path)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                
                Button("关闭") {
                    dismiss()
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
        .frame(width: 400)
    }
}

// 预览
struct AdvancedLogViewer_Previews: PreviewProvider {
    static var previews: some View {
        AdvancedLogViewer()
    }
}