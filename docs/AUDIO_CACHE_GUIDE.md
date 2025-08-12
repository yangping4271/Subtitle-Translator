# 音频缓存使用指南

## 概述

Chrome插件支持智能音频缓存功能，当自动下载失败时，用户可以手动下载音频并上传到缓存，实现无缝的字幕翻译体验。

## 工作原理

### 自动处理流程
1. **插件检测**: 访问YouTube视频时自动提取视频ID
2. **缓存检查**: 优先检查是否已有缓存的音频文件
3. **自动下载**: 无缓存时自动下载（音频优先，视频备用）
4. **缓存存储**: 下载成功后自动缓存到本地
5. **字幕处理**: 使用缓存或下载的音频进行转录翻译

### 手动缓存工作流
当自动下载失败时：
1. **手动下载**: 用户手动下载音频文件
2. **上传缓存**: 通过API上传到后端缓存
3. **刷新页面**: 插件自动使用缓存文件
4. **正常处理**: 跳过下载，直接转录翻译

## 手动下载方法

### 方法1: 使用yt-dlp（推荐）

#### 下载音频（首选）
```bash
# 基础音频下载
yt-dlp --extract-audio --audio-format m4a \
  --cookies-from-browser chrome --geo-bypass \
  -o "~/Downloads/%(id)s.%(ext)s" \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 高质量音频下载
yt-dlp --extract-audio --audio-format m4a --audio-quality 192K \
  --cookies-from-browser chrome --geo-bypass \
  -o "~/Downloads/%(title)s-%(id)s.%(ext)s" \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

#### 下载视频（音频不可用时）
```bash
# 下载低质量视频（节省带宽）
yt-dlp --format "worst[height<=360]/worst" \
  --cookies-from-browser chrome --geo-bypass \
  -o "~/Downloads/%(id)s.%(ext)s" \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 下载视频并提取音频
yt-dlp --format "worst[height<=480]/worst" --extract-audio \
  --audio-format m4a --cookies-from-browser chrome --geo-bypass \
  -o "~/Downloads/%(id)s.%(ext)s" \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### 方法2: 使用其他下载工具

#### 桌面应用
- **4K Video Downloader**: 支持音频/视频分别下载
- **JDownloader**: 支持批量下载和格式选择
- **youtube-dl-gui**: yt-dlp的图形界面版本

#### 在线工具
- **y2mate.com**: 在线YouTube下载器
- **keepvid.com**: 支持多种格式下载
- **savefrom.net**: 简单快速的在线下载

**注意**: 确保下载为支持的音频格式：
- 推荐: `.m4a`, `.mp3`
- 支持: `.wav`, `.aac`, `.ogg`

## 缓存管理API

### 上传音频到缓存
```bash
# 基本上传（替换VIDEO_ID为实际YouTube视频ID）
curl -X POST "http://127.0.0.1:9009/cache/upload_audio/dQw4w9WgXcQ" \
  -F "file=@/path/to/audio.m4a"

# 上传并查看结果
curl -X POST "http://127.0.0.1:9009/cache/upload_audio/dQw4w9WgXcQ" \
  -F "file=@~/Downloads/audio.m4a" | jq .
```

### 检查缓存状态
```bash
# 查看指定视频缓存
curl "http://127.0.0.1:9009/cache/check/dQw4w9WgXcQ" | jq .

# 查看所有缓存状态
curl "http://127.0.0.1:9009/cache/status" | jq .

# 查看缓存统计
curl "http://127.0.0.1:9009/cache/status" | jq '.audio_cache'
```

### 清理缓存
```bash
# 清除指定视频缓存
curl -X DELETE "http://127.0.0.1:9009/cache/clear/dQw4w9WgXcQ"

# 清除所有缓存（手动操作）
rm -rf /tmp/yt_cache/audio/*
rm -rf /tmp/yt_cache/subtitles/*
```

## 实际使用示例

### 示例1: 完整工作流
```bash
# 1. 视频URL
VIDEO_URL="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VIDEO_ID="dQw4w9WgXcQ"

# 2. 检查是否已缓存
curl "http://127.0.0.1:9009/cache/check/$VIDEO_ID"

# 3. 手动下载音频（如果需要）
yt-dlp --extract-audio --audio-format m4a \
  --cookies-from-browser chrome --geo-bypass \
  -o "~/Downloads/$VIDEO_ID.%(ext)s" "$VIDEO_URL"

# 4. 上传到缓存
curl -X POST "http://127.0.0.1:9009/cache/upload_audio/$VIDEO_ID" \
  -F "file=@$HOME/Downloads/$VIDEO_ID.m4a"

# 5. 验证缓存
curl "http://127.0.0.1:9009/cache/check/$VIDEO_ID" | jq .has_audio_cache

# 6. 刷新YouTube页面使用缓存
```

### 示例2: 批量处理
```bash
# 批量下载多个视频的音频
VIDEO_IDS=("dQw4w9WgXcQ" "jNQXAC9IVRw" "y6120QOlsfU")

for video_id in "${VIDEO_IDS[@]}"; do
  echo "处理视频: $video_id"
  
  # 检查缓存
  if curl -s "http://127.0.0.1:9009/cache/check/$video_id" | jq -r .has_audio_cache | grep -q true; then
    echo "  ✅ 已有缓存，跳过"
    continue
  fi
  
  # 下载音频
  echo "  📥 下载音频..."
  yt-dlp --extract-audio --audio-format m4a \
    --cookies-from-browser chrome --geo-bypass \
    -o "~/Downloads/$video_id.%(ext)s" \
    "https://www.youtube.com/watch?v=$video_id"
  
  # 上传到缓存
  if [ -f "$HOME/Downloads/$video_id.m4a" ]; then
    echo "  📤 上传到缓存..."
    curl -X POST "http://127.0.0.1:9009/cache/upload_audio/$video_id" \
      -F "file=@$HOME/Downloads/$video_id.m4a"
    echo "  ✅ 完成"
  else
    echo "  ❌ 下载失败"
  fi
done
```

## 缓存目录结构

```
/tmp/yt_cache/
├── audio/                    # 音频缓存目录
│   ├── dQw4w9WgXcQ.m4a      # 视频ID.扩展名
│   ├── jNQXAC9IVRw.mp3      # 支持多种音频格式
│   └── y6120QOlsfU.wav      # 
├── subtitles/               # 字幕缓存目录
│   ├── dQw4w9WgXcQ.vtt      # YouTube原生字幕
│   └── jNQXAC9IVRw.srt      # 
└── metadata.json           # 缓存元数据（虚拟，存储在内存）
```

## 故障排除

### 常见问题

**1. 上传失败: "不支持的文件类型"**
```bash
# 检查文件格式
file ~/Downloads/audio.m4a

# 转换格式（如果需要）
ffmpeg -i input.webm -acodec copy output.m4a
```

**2. 缓存未生效**
```bash
# 验证上传成功
curl "http://127.0.0.1:9009/cache/check/VIDEO_ID" | jq .has_audio_cache

# 清除浏览器缓存并刷新页面
# 检查插件是否重新检测缓存
```

**3. 下载失败: 地域限制**
```bash
# 使用代理下载
yt-dlp --proxy socks5://127.0.0.1:1080 \
  --extract-audio --audio-format m4a \
  --cookies-from-browser chrome \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

**4. 权限问题**
```bash
# 检查缓存目录权限
ls -la /tmp/yt_cache/

# 修复权限（如果需要）
chmod 755 /tmp/yt_cache/
chmod 755 /tmp/yt_cache/audio/
```

### 调试方法

```bash
# 1. 检查后端服务状态
curl "http://127.0.0.1:9009/health"

# 2. 查看详细缓存信息
curl "http://127.0.0.1:9009/cache/status" | jq .

# 3. 测试文件上传
curl -X POST "http://127.0.0.1:9009/cache/upload_audio/test" \
  -F "file=@test.m4a" -v

# 4. 查看插件日志
# 在YouTube页面按Ctrl+L导出调试日志
```

## 高级技巧

### 1. 自动化脚本
创建一个自动化脚本，当检测到插件下载失败时自动手动下载：

```bash
#!/bin/bash
# auto_cache.sh - 自动缓存脚本

monitor_downloads() {
  local video_url="$1"
  local video_id=$(echo "$video_url" | grep -oP '(?<=v=)[^&]*')
  
  echo "监控视频: $video_id"
  
  # 等待一段时间让自动下载尝试
  sleep 30
  
  # 检查是否成功
  if ! curl -s "http://127.0.0.1:9009/cache/check/$video_id" | jq -r .has_audio_cache | grep -q true; then
    echo "自动下载失败，启动手动下载..."
    manual_download "$video_url" "$video_id"
  fi
}

manual_download() {
  local video_url="$1"
  local video_id="$2"
  
  # 手动下载
  yt-dlp --extract-audio --audio-format m4a \
    --cookies-from-browser chrome --geo-bypass \
    -o "/tmp/$video_id.%(ext)s" "$video_url"
  
  # 上传缓存
  if [ -f "/tmp/$video_id.m4a" ]; then
    curl -X POST "http://127.0.0.1:9009/cache/upload_audio/$video_id" \
      -F "file=@/tmp/$video_id.m4a"
    rm "/tmp/$video_id.m4a"  # 清理临时文件
  fi
}

# 使用方法：./auto_cache.sh "https://www.youtube.com/watch?v=VIDEO_ID"
monitor_downloads "$1"
```

### 2. 预缓存策略
为经常观看的频道或播放列表预缓存音频：

```bash
# 获取播放列表中的所有视频ID
yt-dlp --flat-playlist --get-id "PLAYLIST_URL" > video_ids.txt

# 批量预缓存
while read video_id; do
  echo "预缓存: $video_id"
  yt-dlp --extract-audio --audio-format m4a \
    --cookies-from-browser chrome --geo-bypass \
    -o "/tmp/$video_id.%(ext)s" \
    "https://www.youtube.com/watch?v=$video_id"
  
  if [ -f "/tmp/$video_id.m4a" ]; then
    curl -X POST "http://127.0.0.1:9009/cache/upload_audio/$video_id" \
      -F "file=@/tmp/$video_id.m4a"
    rm "/tmp/$video_id.m4a"
  fi
done < video_ids.txt
```

通过这些方法，你可以确保Chrome插件始终有音频文件可用，实现流畅的双语字幕翻译体验。