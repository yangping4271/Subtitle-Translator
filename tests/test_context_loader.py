from pathlib import Path
from subtitle_translator.context_loader import (
    build_context_info,
    extract_folder_path,
    extract_path_context_titles,
)


def test_extract_folder_path_max_depth():
    path = Path("/a/b/c/d")
    result = extract_folder_path(path, max_depth=3)
    assert result == "b / c / d"


def test_extract_folder_path_replaces_separators():
    path = Path("/root/my_show/season-1")
    result = extract_folder_path(path, max_depth=2)
    assert result == "my show / season 1"


def test_extract_path_context_titles_ignores_system_noise():
    path = Path("/Volumes/T7/Claude Code for Professional Developers")
    result = extract_path_context_titles(path, max_depth=3)
    assert result == ["Claude Code for Professional Developers"]


def test_build_context_info_with_context_file(tmp_path):
    ctx_file = tmp_path / "context.txt"
    ctx_file.write_text("A show about science.", encoding="utf-8")

    srt_file = tmp_path / "episode_01.srt"
    srt_file.touch()

    result = build_context_info(srt_file.resolve())
    assert "A show about science." in result
    assert "episode 01" in result


def test_build_context_info_without_context_file(tmp_path):
    srt_file = tmp_path / "my-film.srt"
    srt_file.touch()

    result = build_context_info(srt_file.resolve())
    assert "my film" in result
    assert "context.txt" not in result


def test_build_context_info_uses_series_and_sibling_titles(tmp_path):
    course_dir = tmp_path / "Claude Code for Professional Developers"
    course_dir.mkdir()

    current_file = course_dir / "1.1- Welcome.en.srt"
    current_file.touch()
    (course_dir / "1.2- What is Claude Code.en.srt").touch()
    (course_dir / "1.3- Installing Claude Code.mp4").touch()
    (course_dir / "1.4- Course Structure.zh-CN.srt").touch()
    (course_dir / "._1.5- Hidden Noise.en.srt").touch()

    result = build_context_info(current_file.resolve())

    assert "Claude Code for Professional Developers" in result
    assert "Sequence label: 1.1" in result
    assert "Current title: Welcome" in result
    assert "High-confidence canonical names:" in result
    assert "- Claude Code" in result
    assert "Nearby titles in the same folder:" in result
    assert "- What is Claude Code" in result
    assert "- Installing Claude Code" in result
    assert "Hidden Noise" not in result
