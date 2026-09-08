"""CLI 基础行为测试（不需要真实 API）"""
from unittest.mock import patch

from typer.testing import CliRunner

from subtitle_translator.cli import app
from subtitle_translator import env_setup

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "字幕翻译" in result.output


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Subtitle Translator" in result.output


def test_version_subcommand():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0


def test_translation_run_starts_a_new_log_task(tmp_path):
    test_srt = tmp_path / "test.srt"
    test_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest subtitle\n")

    with patch("subtitle_translator.cli.setup_environment"), \
         patch("subtitle_translator.cli.start_task_logging") as start_logging, \
         patch("subtitle_translator.cli.process_batch"):
        result = runner.invoke(app, ["-i", str(test_srt)])

    assert result.exit_code == 0
    start_logging.assert_called_once_with()


def test_invalid_language():
    with patch("subtitle_translator.cli.setup_environment"):
        result = runner.invoke(app, ["-t", "xyz_not_a_lang"])
    assert result.exit_code == 1


def test_missing_config(tmp_path, monkeypatch):
    monkeypatch.setattr(env_setup, "_env_loaded", False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SPLIT_MODEL", raising=False)
    monkeypatch.delenv("TRANSLATION_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with patch("subtitle_translator.env_setup._get_config_path", return_value=tmp_path / ".env"):
        result = runner.invoke(app, ["--dry-run"])
    assert result.exit_code == 1


def test_missing_models(tmp_path, monkeypatch):
    monkeypatch.setattr(env_setup, "_env_loaded", False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("SPLIT_MODEL", raising=False)
    monkeypatch.delenv("TRANSLATION_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with patch("subtitle_translator.env_setup._get_config_path", return_value=tmp_path / ".env"):
        result = runner.invoke(app, ["--dry-run"])
    assert result.exit_code == 1
    assert "SPLIT_MODEL" in result.output
    assert "TRANSLATION_MODEL" in result.output


def test_dry_run_allows_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(env_setup, "_env_loaded", False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SPLIT_MODEL", "test-split")
    monkeypatch.setenv("TRANSLATION_MODEL", "test-translation")

    test_srt = tmp_path / "test.srt"
    test_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest subtitle\n")

    result = runner.invoke(app, ["--dry-run", "-i", str(test_srt)])
    assert result.exit_code == 0


def test_dry_run_rejects_loopback_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(env_setup, "_env_loaded", False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.2:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    test_srt = tmp_path / "test.srt"
    test_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest subtitle\n")

    result = runner.invoke(app, ["--dry-run", "-i", str(test_srt)])
    assert result.exit_code == 1
    assert "不支持本地模型服务" in result.output


def test_init_rejects_loopback_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with patch(
        "rich.prompt.Prompt.ask",
        side_effect=["http://localhost:1234/v1", "test-key"],
    ):
        result = runner.invoke(app, ["init"])

    assert result.exit_code == 1
    assert "不支持本地模型服务" in result.output


def test_init_writes_explicit_model_names(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with patch(
        "rich.prompt.Prompt.ask",
        side_effect=[
            "https://api.openai.com/v1",
            "test-key",
            "my-split-model",
            "my-translation-model",
        ],
    ):
        result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    config_text = (tmp_path / ".config" / "subtitle-translator" / ".env").read_text()
    assert "SPLIT_MODEL=my-split-model" in config_text
    assert "TRANSLATION_MODEL=my-translation-model" in config_text
    assert "gpt-4o" not in config_text
    assert "LLM_MODEL=" not in config_text


def test_dry_run_empty_dir(tmp_path):
    """空目录应该报错：没有找到 SRT 文件"""
    with patch("subtitle_translator.cli.setup_environment"):
        result = runner.invoke(app, ["--dry-run", "--input-dir", str(tmp_path)])
    assert result.exit_code == 1  # 应该失败
    assert "没有找到" in result.output or "No" in result.output  # 应该有错误提示


def test_dry_run_does_not_create_output_dir(tmp_path):
    """测试 dry-run 模式不会创建输出目录"""
    # 创建一个测试 SRT 文件
    test_srt = tmp_path / "test.srt"
    test_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest subtitle\n")

    # 指定一个不存在的输出目录
    output_dir = tmp_path / "output_should_not_exist"

    with patch("subtitle_translator.cli.setup_environment"):
        result = runner.invoke(app, [
            "--dry-run",
            "-i", str(test_srt),
            "-o", str(output_dir)
        ])

    # dry-run 应该成功退出
    assert result.exit_code == 0
    # 输出目录不应该被创建
    assert not output_dir.exists()
