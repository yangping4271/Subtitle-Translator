# 双语字幕解决方案 - 完整指南

本文档介绍了如何使用项目的双语字幕功能，包括命令行工具和 Chrome 扩展两种使用方式。

## 🎯 功能概述

项目现在支持完整的双语字幕解决方案，包含三种使用模式：

1. **ASS 双语字幕文件** - 主项目生成的专业双语字幕（适用于本地播放器）
2. **SRT 中间文件** - 保留的英文和翻译SRT文件（便于二次开发）
3. **Chrome 扩展实时显示** - YouTube浏览器内实时双语字幕（配合智能缓存系统）

### ✅ 已完成功能状态
- **命令行工具**: `--preserve-intermediate` 选项已实现
- **后端服务**: 完整的FastAPI服务支持YouTube视频处理
- **智能缓存**: 三级缓存系统（音频、字幕、翻译）大幅提升处理速度
- **Chrome扩展**: SRT双语模式 + Segments回退模式双重保障
- **错误处理**: 完善的异常处理和超时机制

## 📋 方案1: ASS 双语字幕文件（推荐）

### 特点
- ✅ 专业的双语字幕格式，支持样式和定位
- ✅ 兼容大多数播放器（VLC、PotPlayer、MPC-HC等）
- ✅ 英文原文（青色，上方）+ 中文翻译（绿色，下方）
- ✅ 可自定义字体和颜色

### 使用方法

```bash
# 基本用法：生成双语ASS文件
translate -i video.mp4 -t zh

# 生成的文件：
# - video.srt     (原始转录)
# - video.ass     (双语字幕文件)
```

### ASS 文件结构

```
[V4+ Styles]
Style: Default,Noto Serif,18,&H0000FFFF,...     # 英文样式（青色）
Style: Secondary,宋体-简 黑体,11,&H0000FF00,... # 中文样式（绿色）

[Events]
Dialogue: 0,00:00:01.000,00:00:05.000,Secondary,,0,0,0,,你好，世界！
Dialogue: 0,00:00:01.000,00:00:05.000,Default,,0,0,0,,Hello world!
```

## 📋 方案2: SRT 中间文件保留（新功能）

### 特点
- ✅ 标准 SRT 格式，通用性强
- ✅ 可分别获取英文和翻译的 SRT 文件
- ✅ 便于进一步处理和自定义
- ✅ Chrome 扩展可直接使用

### 使用方法

```bash
# 启用中间文件保留
translate -i video.mp4 -t zh --preserve-intermediate

# 生成的文件：
# - video.srt      (原始转录)
# - video.en.srt   (英文字幕)  ← 新增
# - video.zh.srt   (中文翻译) ← 新增  
# - video.ass      (双语ASS文件)
```

### 支持的语言代码

| 语言 | 代码 | 生成文件示例 |
|------|------|--------------|
| 简体中文 | zh | video.zh.srt |
| 繁体中文 | zh-tw | video.zh-tw.srt |
| 日文 | ja | video.ja.srt |
| 韩文 | ko | video.ko.srt |
| 法文 | fr | video.fr.srt |

### SRT 文件内容示例

**英文 SRT (video.en.srt):**
```
1
00:00:01,000 --> 00:00:05,000
Hello world, this is a test.

2
00:00:06,000 --> 00:00:10,000
This is a simple subtitle test file.
```

**中文 SRT (video.zh.srt):**
```
1
00:00:01,000 --> 00:00:05,000
你好，世界，这是一个测试。

2
00:00:06,000 --> 00:00:10,000
这是一个简单的字幕测试文件。
```

## 📋 方案3: Chrome 扩展实时显示

### 特点
- ✅ 支持 SRT 双语模式（新增）
- ✅ 支持实时翻译模式（原有）
- ✅ 自动检测并使用保留的 SRT 文件
- ✅ 无需手动文件操作

### SRT 双语模式工作流程

1. **后端处理**：使用 `--preserve-intermediate` 生成 SRT 文件
2. **文件检测**：Chrome 扩展检查作业完成状态
3. **SRT 获取**：通过 `/srt_files/{job_id}` API 获取双语 SRT
4. **文件解析**：使用 SRT 解析器处理文件内容
5. **双语显示**：实时同步显示英文原文和中文翻译

### 新增 API 端点

```javascript
// 获取作业的双语 SRT 文件
GET /srt_files/{job_id}

// 响应格式：
{
  "job_id": "abc123",
  "video_id": "dQw4w9WgXcQ", 
  "has_english_srt": true,
  "has_translated_srt": true,
  "english_srt": "1\n00:00:01,000 --> 00:00:05,000\nHello world...",
  "translated_srt": "1\n00:00:01,000 --> 00:00:05,000\n你好，世界..."
}
```

### SRT 解析器功能

新增的 `chrome-extension/srt-parser.js` 提供：

```javascript
// 解析 SRT 文件
const subtitles = SRTParser.parse(srtContent);

// 查找时间点对应的字幕
const current = SRTParser.findSubtitleAtTime(subtitles, currentTime);

// 合并双语字幕
const bilingual = SRTParser.mergeBilingualSubtitles(englishSubs, translatedSubs);

// 验证 SRT 格式
const validation = SRTParser.validate(srtContent);
```

## 🚀 完整使用示例

### 1. 命令行处理

```bash
# 第一步：配置 API 密钥
translate init

# 第二步：处理视频并保留中间文件
translate -i my_video.mp4 -t zh --preserve-intermediate --reflect

# 输出文件：
# - my_video.srt      (原始转录，保留)
# - my_video.en.srt   (英文字幕，新增)
# - my_video.zh.srt   (中文翻译，新增)
# - my_video.ass      (双语ASS文件)
```

### 2. Chrome 扩展使用

```bash
# 第一步：启动后端服务
uv run python backend/server.py

# 第二步：访问 YouTube 视频
# Chrome 扩展会自动：
# 1. 检测视频变化
# 2. 调用后端转录和翻译
# 3. 优先获取 SRT 文件
# 4. 实现实时双语显示
```

### 3. 播放器使用 ASS 文件

**VLC 播放器：**
1. 打开视频文件
2. 字幕 → 添加字幕文件 → 选择 `.ass` 文件
3. 享受双语字幕体验

**PotPlayer：**
1. 右键 → 字幕 → 选择字幕
2. 选择对应的 `.ass` 文件
3. 自动显示双语字幕

## 🔧 技术实现细节

### 核心修改

1. **处理器增强** (`src/subtitle_translator/processor.py`)
   - 添加 `preserve_intermediate` 参数
   - 修改清理逻辑，选择性保留 SRT 文件

2. **CLI 扩展** (`src/subtitle_translator/cli.py`)
   - 新增 `--preserve-intermediate` / `-p` 选项
   - 传递参数到处理函数

3. **后端 API** (`backend/server.py`)
   - 新增 `/srt_files/{job_id}` 端点
   - 自动使用 `--preserve-intermediate` 选项
   - 智能查找和返回 SRT 文件

4. **Chrome 扩展增强**
   - 新增 `srt-parser.js` SRT 解析工具
   - 实现双语 SRT 模式
   - 优化文件获取和显示逻辑

### 文件命名规则

```
基础名称: video
英文 SRT: video.en.srt
中文 SRT: video.zh.srt
日文 SRT: video.ja.srt
双语 ASS: video.ass
```

### 后端文件搜索逻辑

```python
# 搜索路径优先级
search_paths = [
    f"/tmp/yt_subs/*{video_id}*.srt",    # 主输出目录
    f"/tmp/*{video_id}*.srt",            # 临时目录
    f"./*{video_id}*.srt"                # 当前目录
]

# 文件识别规则
english_srt = files_with(".en.srt")
translated_srt = files_with([".zh.srt", ".ja.srt", ".ko.srt"])
```

## 📊 对比总结

| 特性 | ASS 双语文件 | SRT 中间文件 | Chrome 扩展 |
|------|--------------|--------------|-------------|
| 使用场景 | 本地播放器 | 开发/自定义 | YouTube 在线 |
| 双语支持 | ✅ 原生支持 | ✅ 分离文件 | ✅ 实时合并 |
| 样式控制 | ✅ 丰富样式 | ❌ 无样式 | ✅ CSS 样式 |
| 通用性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 开发便利 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🚀 性能优势

### 处理速度对比

**首次处理（无缓存）**：
- 下载音频：1-3 分钟
- 转录处理：3-8 分钟  
- 翻译处理：2-5 分钟
- 总时间：**6-16 分钟**

**翻译缓存命中** 🚀：
- 检测缓存：< 1 秒
- 加载SRT文件：< 5 秒
- 总时间：**几秒钟内完成**

**音频缓存命中**：
- 跳过下载步骤：节省 1-3 分钟
- 转录+翻译：5-10 分钟
- 总时间：**5-10 分钟**

**使用中间文件**：
- 跳过转录步骤
- 直接处理现有 SRT 文件
- 总时间：**2-5 分钟**

### 缓存效果
- **音频缓存**: 避免重复下载，节省带宽和时间
- **字幕缓存**: YouTube原始字幕辅助翻译优化
- **翻译缓存**: 完整翻译结果缓存，实现秒级加载
- **智能管理**: 自动缓存清理和完整性验证

## 🎉 总结

通过这次实现，我们为项目添加了强大的双语字幕解决方案：

1. **保持向后兼容** - 原有的 ASS 双语文件功能不变
2. **增强开发体验** - 新的 SRT 中间文件便于进一步处理
3. **优化 Chrome 扩展** - 支持更高质量的双语显示方式，配合智能缓存系统
4. **提供多种选择** - 根据使用场景选择最适合的方案
5. **大幅提升性能** - 智能缓存系统将重复访问时间从分钟级别降低到秒级别

无论是本地播放器用户、开发者还是 Chrome 扩展用户，都能找到适合的双语字幕解决方案！