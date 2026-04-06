"""
上下文加载模块 - 负责提取和加载翻译上下文信息
"""
from pathlib import Path


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

    readable_filename = input_file.stem.replace('_', ' ').replace('-', ' ').strip()
    if readable_filename:
        hints.append(readable_filename)

    current_path = input_file.parent
    parent_names = []
    for _ in range(max_depth):
        if not current_path.name or current_path.name in ['/', '.', '..']:
            break
        folder_name = current_path.name.replace('_', ' ').replace('-', ' ').strip()
        if folder_name:
            parent_names.append(folder_name)
        current_path = current_path.parent

    hints.extend(parent_names)
    return hints
