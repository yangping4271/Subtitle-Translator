"""
上下文加载模块 - 负责提取和加载翻译上下文信息
"""
from pathlib import Path


def build_context_info(input_file: Path) -> str:
    """构建完整的上下文信息：外部文件 + 文件名/路径"""
    context_parts = []

    external_context = read_external_context(input_file.parent)
    if external_context:
        context_parts.append(external_context)

    folder_path = extract_folder_path(input_file.parent)
    if folder_path:
        context_parts.append(f"Folder path: {folder_path}")

    readable_filename = input_file.stem.replace('_', ' ').replace('-', ' ')
    context_parts.append(f"Filename: {readable_filename}")

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
