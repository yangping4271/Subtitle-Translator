# Install And Discovery

这个 skill 依赖外部工具 `Subtitle-Translator`，因此需要明确区分：

- 安装的是 skill 本身
- 安装的是 skill 依赖的 CLI 工具

## 发现顺序

1. `translate` 是否已在 `PATH`
2. 是否存在可用本地仓库
3. 若都没有，再克隆官方仓库并执行 `uv tool install -e .`

## 脚本职责

- `scripts/discover_subtitle_translator.sh`
  - 返回 `mode=cli`、`mode=repo` 或 `mode=missing`
- `scripts/install_subtitle_translator.sh`
  - 安装或重装 Subtitle-Translator
- `scripts/run_translate.sh`
  - 统一执行入口，自动选择 `translate` 或 `uv run`

## 安装后最小验证

```bash
translate --help
translate --version
```
