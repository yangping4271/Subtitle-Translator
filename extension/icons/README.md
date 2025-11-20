# 图标创建说明

我已经创建了一个SVG图标 (`icon.svg`)，您需要将其转换为PNG格式：

## 图标设计
- 蓝色圆形背景（代表YouTube）
- "CC" 字样（代表字幕/Closed Caption）
- 底部字幕框样式
- 彩色装饰点

## 转换为PNG的方法

### 方法1：在线转换工具
1. 访问 https://www.svg2png.com/ 或 https://svgtopng.com/
2. 上传 `icon.svg` 文件
3. 生成以下尺寸：
   - 16x16 像素 → 保存为 `icon16.png`
   - 48x48 像素 → 保存为 `icon48.png`  
   - 128x128 像素 → 保存为 `icon128.png`

### 方法2：使用设计软件
- Adobe Illustrator / Photoshop
- Figma (免费)
- GIMP (免费)
- Inkscape (免费)

### 方法3：使用命令行工具 (如果已安装)
```bash
# 使用ImageMagick
magick icon.svg -resize 16x16 icon16.png
magick icon.svg -resize 48x48 icon48.png
magick icon.svg -resize 128x128 icon128.png

# 使用rsvg-convert
rsvg-convert -w 16 -h 16 icon.svg > icon16.png
rsvg-convert -w 48 -h 48 icon.svg > icon48.png
rsvg-convert -w 128 -h 128 icon.svg > icon128.png
```

## 放置位置
将生成的PNG文件放在 `extension/icons/` 文件夹中：
- extension/icons/icon16.png
- extension/icons/icon48.png
- extension/icons/icon128.png