// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SubtitleTranslatorApp",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "SubtitleTranslatorApp",
            targets: ["SubtitleTranslatorApp"]
        ),
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "SubtitleTranslatorApp",
            dependencies: [],
            path: "SubtitleTranslatorApp",
            exclude: ["Resources", "Info.plist"],
            sources: [
                "App.swift",
                "Views/ContentView.swift", 
                "Views/SidebarView.swift",
                "Views/DropZoneView.swift",
                "Views/ProcessingQueueView.swift",
                "Models/ProcessingModels.swift",
                "Services/ProcessingManager.swift",
                "Services/PythonBridgeService.swift",
                "Services/ConfigurationService.swift",
                "Services/AppLogger.swift"
            ]
        ),
    ]
)