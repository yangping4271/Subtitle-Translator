import SwiftUI

@main
struct SubtitleTranslatorApp: App {
    init() {
        // 初始化日志系统
        appLog("===== Subtitle Translator App 启动 =====")
        appLog("版本: 1.0.0")
        appLog("系统: macOS \(ProcessInfo.processInfo.operatingSystemVersionString)")
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
    }
}