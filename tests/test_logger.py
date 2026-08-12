from pathlib import Path
import logging
import queue

from subtitle_translator import logger as logger_module
from subtitle_translator.logger import (
    QueueListenerHandler,
    _get_log_path,
    _prepare_log_file,
    clear_log_file,
)


def test_log_path_is_stable_and_honors_xdg_data_home(monkeypatch, tmp_path):
    monkeypatch.delenv("SUBTITLE_TRANSLATOR_LOG_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    assert _get_log_path() == tmp_path / "subtitle-translator" / "logs"


def test_explicit_log_directory_takes_priority(monkeypatch, tmp_path):
    log_dir = tmp_path / "custom-logs"
    monkeypatch.setenv("SUBTITLE_TRANSLATOR_LOG_DIR", str(log_dir))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    assert _get_log_path() == log_dir


def test_prepare_log_file_restricts_permissions(tmp_path):
    log_file = tmp_path / "logs" / "app.log"
    log_file.parent.mkdir()
    log_file.write_text("private subtitles", encoding="utf-8")
    log_file.chmod(0o644)

    _prepare_log_file(log_file)

    assert log_file.stat().st_mode & 0o777 == 0o600
    assert Path(log_file.parent).stat().st_mode & 0o777 == 0o700


def test_clear_log_file_truncates_existing_file(tmp_path):
    log_file = tmp_path / "logs" / "app.log"
    log_file.parent.mkdir()
    log_file.write_text("previous task", encoding="utf-8")

    clear_log_file(log_file)

    assert log_file.read_text(encoding="utf-8") == ""
    assert log_file.stat().st_mode & 0o777 == 0o600


def test_queue_listener_clear_keeps_new_records(tmp_path, monkeypatch):
    log_file = tmp_path / "logs" / "app.log"
    monkeypatch.setattr(logger_module, "LOG_FILE", str(log_file))
    handler = QueueListenerHandler(queue.Queue(), logging.DEBUG)
    handler.start_listener()

    try:
        handler.handle(logging.LogRecord(
            "test", logging.INFO, __file__, 1, "old record", (), None
        ))
        handler.queue.join()
        handler.clear_file()
        handler.handle(logging.LogRecord(
            "test", logging.INFO, __file__, 2, "new record", (), None
        ))
        handler.queue.join()
    finally:
        handler._queue_listener.stop()
        handler._file_handler.close()

    assert log_file.read_text(encoding="utf-8").endswith("new record\n")
    assert "old record" not in log_file.read_text(encoding="utf-8")
