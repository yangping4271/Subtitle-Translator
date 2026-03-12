from subtitle_translator.file_discovery import (
    remove_language_suffix,
    natural_sort_key,
    format_file_size,
    get_file_type_info,
)


def test_remove_language_suffix():
    assert remove_language_suffix("movie.zh") == "movie"
    assert remove_language_suffix("movie.zh-cn") == "movie"
    assert remove_language_suffix("movie.zh-tw") == "movie"
    assert remove_language_suffix("movie.ja") == "movie"
    assert remove_language_suffix("movie.ko") == "movie"
    assert remove_language_suffix("movie.fr") == "movie"
    assert remove_language_suffix("movie.en") == "movie"
    assert remove_language_suffix("movie") == "movie"


def test_natural_sort_key():
    files = ["EP10", "EP2", "EP1", "EP20"]
    sorted_files = sorted(files, key=natural_sort_key)
    assert sorted_files == ["EP1", "EP2", "EP10", "EP20"]


def test_format_file_size(tmp_path):
    small = tmp_path / "small.txt"
    small.write_bytes(b"x" * 500)
    assert format_file_size(small) == "500 B"

    kb = tmp_path / "kb.txt"
    kb.write_bytes(b"x" * 2048)
    assert format_file_size(kb) == "2.0 KB"

    mb = tmp_path / "mb.txt"
    mb.write_bytes(b"x" * (2 * 1024 * 1024))
    assert format_file_size(mb) == "2.0 MB"


def test_get_file_type_info():
    file_type, process_type = get_file_type_info(".srt")
    assert file_type == "字幕文件"
    assert process_type == "直接翻译"

    file_type, process_type = get_file_type_info(".mp4")
    assert file_type == "未知类型"
