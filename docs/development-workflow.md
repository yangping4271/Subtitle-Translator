# 开发工作流程

## 核心原则

**测试时首选 `uv run`，确认后再安装**

## 标准流程

```bash
# 1. 安装依赖
uv sync --dev

# 2. 直接运行源码测试（修改立即生效）
uv run python -m subtitle_translator.cli -i test.srt
uv run python -m subtitle_translator.transcription_core.cli test.mp4

# 3. 验证输出
cat test.zh.srt | head -20
cat test.ass | head -30

# 4. 确认无误后，开发模式安装
uv tool install --force -e .

# 5. 使用安装的命令
translate -i your-file.srt
transcribe your-file.mp4
```

## 为什么这样做

- **`uv run` 优势**：无缓存、直接运行源码、易调试
- **`--force -e` 优势**：强制覆盖、链接源码、修改立即生效

## 测试示例

```bash
# 修改代码
vim src/subtitle_translator/translation_core/data.py

# 直接测试
uv run python -m subtitle_translator.cli -i test.srt

# 验证功能
cat test.zh.srt | grep "大家好"

# 测试参数
uv run python -m subtitle_translator.cli -i test.srt --keep-punctuation
```

## 故障排除

```bash
# 彻底清理
uv tool uninstall subtitle-translator
uv cache clean
find src -name "*.pyc" -delete
find src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 重新安装
uv tool install --force -e .
```

## 生产安装

```bash
uv tool install .
```
