"""CLI 基础行为测试（不需要真实 API）"""
from unittest.mock import patch

from typer.testing import CliRunner

from subtitle_translator.cli import app

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


def test_invalid_language():
    with patch("subtitle_translator.cli.setup_environment"):
        result = runner.invoke(app, ["-t", "xyz_not_a_lang"])
    assert result.exit_code == 1


def test_missing_config(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("subtitle_translator.env_setup._get_config_path", return_value=tmp_path / ".env"):
        result = runner.invoke(app, ["--dry-run"])
    assert result.exit_code == 1


def test_dry_run_empty_dir(tmp_path):
    """空目录应该报错：没有找到 SRT 文件"""
    with patch("subtitle_translator.cli.setup_environment"):
        result = runner.invoke(app, ["--dry-run", "--input-dir", str(tmp_path)])
    assert result.exit_code == 1  # 应该失败
    assert "没有找到" in result.output or "No" in result.output  # 应该有错误提示
