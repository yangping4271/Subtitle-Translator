from ..types import ThinkingCapability, ThinkingDisableMethod, ThinkingPlugin


def _is_gemini_25_flash(name: str) -> bool:
    return name.startswith("gemini-2.5-flash")


def _is_gemini_3_or_later(name: str) -> bool:
    if not name.startswith("gemini-"):
        return False
    version = name.removeprefix("gemini-")
    if version.startswith("2.5"):
        return False
    major = version.split(".", 1)[0].split("-", 1)[0]
    return major.isdigit() and int(major) >= 3


PLUGINS = (
    ThinkingPlugin(
        match=_is_gemini_25_flash,
        method=ThinkingDisableMethod.GOOGLE_THINKING_BUDGET,
        capability=ThinkingCapability.DISABLED,
    ),
    ThinkingPlugin(
        match=_is_gemini_3_or_later,
        method=ThinkingDisableMethod.GOOGLE_THINKING_LEVEL,
        capability=ThinkingCapability.DISABLED,
    ),
)
