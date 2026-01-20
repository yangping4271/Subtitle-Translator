# 转录模型配置

## 默认模型

`mlx-community/parakeet-tdt-0.6b-v2`（约 1.2GB）

**限制**：仅支持英语音频/视频

## 下载模型

```bash
hf download mlx-community/parakeet-tdt-0.6b-v2
```

下载位置：
- macOS/Linux: `~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2/`
- 自定义：设置环境变量 `HF_HOME`

## 使用其他模型

**配置文件**：
- 开发模式：项目根目录 `.env`
- 全局模式：`~/.config/subtitle-translator/.env`

**方式 1：配置文件**

```bash
TRANSCRIPTION_MODEL_PATH=/path/to/your/model
```

**方式 2：命令行参数**

```bash
uv run python -m subtitle_translator.transcription_core.cli audio.mp4 --model /path/to/model
```

**要求**：
- 兼容 Parakeet MLX 格式
- 包含 `config.json` 和 `model.safetensors`
