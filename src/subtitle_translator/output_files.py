"""Translated subtitle file creation and lifecycle."""

from dataclasses import dataclass
from pathlib import Path

from .logger import setup_logger
from .translation_core.data import SubtitleData
from .translation_core.utils.ass_converter import convert_srt_to_ass

logger = setup_logger(__name__)


@dataclass(frozen=True)
class SubtitleOutputFiles:
    """Paths produced for one source subtitle."""

    target_srt: Path
    source_srt: Path
    bilingual_ass: Path
    intermediates_preserved: bool


def write_subtitle_outputs(
    sentence_subtitle: SubtitleData,
    translation_results: list[dict],
    input_srt_path: Path,
    output_dir: Path,
    target_lang: str,
    *,
    preserve_intermediate: bool,
) -> SubtitleOutputFiles:
    """Write both SRT files, build the bilingual ASS, and apply cleanup policy."""
    base_name = input_srt_path.stem
    target_srt = output_dir / f"{base_name}.{target_lang}.srt"
    source_srt = output_dir / f"{base_name}.en.srt"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("翻译文件将保存到目录: %s", output_dir)
    logger.info("目标语言文件: %s", target_srt)
    logger.info("英文文件: %s", source_srt)

    try:
        sentence_subtitle.save_translations_to_files(
            translation_results,
            str(source_srt),
            str(target_srt),
        )

        bilingual_ass = convert_srt_to_ass(target_srt, source_srt, output_dir)

        logger.info("双语 ASS 文件: %s", bilingual_ass)
        return SubtitleOutputFiles(
            target_srt=target_srt,
            source_srt=source_srt,
            bilingual_ass=bilingual_ass,
            intermediates_preserved=preserve_intermediate,
        )
    finally:
        if not preserve_intermediate:
            cleaned_files = 0
            for path in (target_srt, source_srt):
                if path.exists():
                    path.unlink()
                    logger.info("已删除中间文件: %s", path)
                    cleaned_files += 1
            if cleaned_files:
                logger.info("已清理 %s 个中间字幕文件", cleaned_files)
