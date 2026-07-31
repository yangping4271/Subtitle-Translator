from pathlib import Path
from subtitle_translator.context_loader import (
    build_context_info,
    extract_folder_path,
    extract_terminology_hints,
)


def test_extract_folder_path_max_depth():
    path = Path("/a/b/c/d")
    result = extract_folder_path(path, max_depth=3)
    assert result == "b / c / d"


def test_extract_folder_path_replaces_separators():
    path = Path("/root/my_show/season-1")
    result = extract_folder_path(path, max_depth=2)
    assert result == "my show / season 1"


def test_build_context_info_with_context_file(tmp_path):
    ctx_file = tmp_path / "context.txt"
    ctx_file.write_text("A show about science.", encoding="utf-8")

    srt_file = tmp_path / "episode_01.srt"
    srt_file.touch()

    result = build_context_info(srt_file.resolve())
    assert "A show about science." in result
    assert "Terminology hints:" in result
    assert "episode 01" in result


def test_build_context_info_without_context_file(tmp_path):
    srt_file = tmp_path / "my-film.srt"
    srt_file.touch()

    result = build_context_info(srt_file.resolve())
    assert "Terminology hints:" in result
    assert "my film" in result
    assert "context.txt" not in result
    assert "Filename:" not in result
    assert "Folder path:" not in result


def test_terminology_hints_filter_generic_paths_and_video_id(monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/Users/yangping")))
    input_file = Path(
        "/Users/yangping/Downloads/"
        "Benchling cuts migration time - Cursor_f_iA2NtIxBU.srt"
    )

    hints = extract_terminology_hints(input_file)

    assert hints == ["Benchling cuts migration time Cursor"]
