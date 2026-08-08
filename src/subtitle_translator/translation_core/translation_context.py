"""Per-file Translation context loading."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import SubtitleConfig, get_target_language
from .external_glossary import load_external_terminology
from .terminology import load_terminology


@dataclass(frozen=True)
class TranslationContext:
    """Immutable target language and terminology for one source subtitle."""

    target_language: str
    terminology: Mapping[str, Any] = field(default_factory=dict)
    external_terminology: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def for_file(
        cls,
        config: SubtitleConfig,
        target_lang: str,
        input_srt_path: Path,
    ) -> "TranslationContext":
        """Load all per-file translation state behind one interface."""
        target_language = get_target_language(target_lang)
        terminology = load_terminology(target_language, input_srt_path)
        external_terminology = {}
        if config.external_glossary_enabled:
            external_terminology = load_external_terminology(
                target_language,
                config.external_glossary_domains,
            )
        return cls(
            target_language=target_language,
            terminology=terminology,
            external_terminology=external_terminology,
        )
