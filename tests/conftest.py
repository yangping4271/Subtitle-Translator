import os
import tempfile
from pathlib import Path


_test_log_dir = Path(tempfile.gettempdir()) / (
    f"subtitle-translator-tests-{os.getpid()}"
)
os.environ["SUBTITLE_TRANSLATOR_LOG_DIR"] = str(_test_log_dir)
