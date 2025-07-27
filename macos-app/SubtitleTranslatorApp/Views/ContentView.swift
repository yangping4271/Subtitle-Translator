import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var processingManager = ProcessingManager()
    @State private var isDragOver = false
    
    var body: some View {
        HSplitView {
            // 左侧边栏
            SidebarView(processingManager: processingManager)
                .frame(minWidth: 250, maxWidth: 350)
            
            // 主内容区域
            VStack(spacing: 20) {
                // 拖拽区域
                DropZoneView(isDragOver: $isDragOver, processingManager: processingManager)
                    .frame(maxHeight: 200)
                
                // 处理队列
                ProcessingQueueView(processingManager: processingManager)
                
                Spacer()
            }
            .padding()
        }
        .frame(minWidth: 800, minHeight: 600)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button("设置") {
                    // 打开设置窗口
                }
                .buttonStyle(.bordered)
                
                Button("处理") {
                    processingManager.startProcessing()
                }
                .buttonStyle(.borderedProminent)
                .disabled(processingManager.processingQueue.isEmpty)
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}