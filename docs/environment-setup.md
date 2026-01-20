# 环境配置

## 配置文件位置

**全局模式**（推荐）：
- 配置：`~/.config/subtitle-translator/.env`
- 日志：`~/.local/share/subtitle-translator/logs/app.log`

**开发模式**：
- 配置：项目根目录 `.env`
- 日志：项目根目录 `logs/app.log`

## 首次使用

```bash
# 1. 下载转录模型（必需）
hf download mlx-community/parakeet-tdt-0.6b-v2

# 2. 初始化翻译配置（可选）
uv run python -m subtitle_translator.cli init

# 3. 测试
uv run python -m subtitle_translator.transcription_core.cli audio.mp4
uv run python -m subtitle_translator.cli -i test.srt
```

## 手动配置

### 翻译配置

```bash
mkdir -p ~/.config/subtitle-translator

cat > ~/.config/subtitle-translator/.env << 'EOF'
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-key
SPLIT_MODEL=gpt-4o-mini
TRANSLATION_MODEL=gpt-4o
SUMMARY_MODEL=gpt-4o-mini
LLM_MODEL=gpt-4o-mini
EOF

chmod 600 ~/.config/subtitle-translator/.env
```

### 转录模型配置（可选）

默认自动加载 `mlx-community/parakeet-tdt-0.6b-v2`。

使用其他模型：

```bash
echo "TRANSCRIPTION_MODEL_PATH=/path/to/your/model" >> ~/.config/subtitle-translator/.env
```

或命令行指定：

```bash
uv run python -m subtitle_translator.transcription_core.cli audio.mp4 --model /path/to/model
```
