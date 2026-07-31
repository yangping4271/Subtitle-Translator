"""
上下文加载模块 - 负责提取和加载翻译上下文信息
"""
import re
from pathlib import Path


GENERIC_FOLDER_NAMES = {
    "desktop",
    "documents",
    "downloads",
    "home",
    "movies",
    "private",
    "temp",
    "tmp",
    "users",
    "var",
    "videos",
    "volumes",
}


def _readable_path_name(name: str) -> str:
    """清理路径名中的分隔符和常见视频 ID 后缀。"""
    name = re.sub(r"[_-][A-Za-z0-9_-]{11}$", "", name)
    return " ".join(re.sub(r"[_-]+", " ", name).split())


def _is_useful_folder(path: Path) -> bool:
    """过滤系统目录、通用目录和当前用户主目录。"""
    name = _readable_path_name(path.name)
    if not name or name.casefold() in GENERIC_FOLDER_NAMES:
        return False
    try:
        if path.resolve() == Path.home().resolve():
            return False
    except OSError:
        pass
    return True


def build_context_info(input_file: Path) -> str:
    """构建完整的上下文信息：外部文件 + 文件系统术语参考"""
    context_parts = []

    external_context = read_external_context(input_file.parent)
    if external_context:
        context_parts.append(external_context)

    terminology_hints = extract_terminology_hints(input_file)
    if terminology_hints:
        hint_lines = ["Terminology hints:"]
        hint_lines.extend(f"- {hint}" for hint in terminology_hints)
        context_parts.append("\n".join(hint_lines))

    return "\n\n".join(context_parts)


def read_external_context(parent_dir: Path) -> str:
    """读取外部上下文文件"""
    for ctx_filename in ['context.txt', 'ctx.txt']:
        ctx_file = parent_dir / ctx_filename
        if ctx_file.exists():
            try:
                content = ctx_file.read_text(encoding='utf-8').strip()
                if content:
                    return content
            except Exception:
                pass
    return ""


def extract_folder_path(parent_dir: Path, max_depth: int = 3) -> str:
    """提取文件夹路径信息"""
    parent_names = []
    current_path = parent_dir

    for _ in range(max_depth):
        if not current_path.name or current_path.name in ['/', '.', '..']:
            break
        folder_name = current_path.name.replace('_', ' ').replace('-', ' ')
        parent_names.append(folder_name)
        current_path = current_path.parent

    return ' / '.join(reversed(parent_names))


def extract_terminology_hints(input_file: Path, max_depth: int = 3) -> list[str]:
    """提取文件名和上层目录名，作为术语参考。"""
    hints = []

    readable_filename = _readable_path_name(input_file.stem)
    if readable_filename:
        hints.append(readable_filename)

    current_path = input_file.parent
    for _ in range(max_depth):
        if not current_path.name or current_path.name in ['/', '.', '..']:
            break
        if _is_useful_folder(current_path):
            folder_name = _readable_path_name(current_path.name)
            if folder_name and all(
                folder_name.casefold() != hint.casefold() for hint in hints
            ):
                hints.append(folder_name)
        current_path = current_path.parent

    return hints
