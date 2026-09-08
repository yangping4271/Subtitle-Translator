"""思考适配的公共类型。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class ThinkingDisableMethod(str, Enum):
    """已知可用的关闭思考方式。"""

    OPENAI_REASONING_EFFORT = "openai_reasoning_effort"
    THINKING_TYPE_DISABLED = "thinking_type_disabled"
    OPENROUTER_REASONING = "openrouter_reasoning"
    GOOGLE_THINKING_LEVEL = "google_thinking_level"
    GOOGLE_THINKING_BUDGET = "google_thinking_budget"
    ENABLE_THINKING_FALSE = "enable_thinking_false"


class ThinkingCapability(str, Enum):
    """一次请求实际采取的思考策略。"""

    DISABLED = "disabled"
    DEFAULT = "default"
    FEATURE_OFF = "feature_off"

    def console_suffix(self) -> str:
        if self is ThinkingCapability.DISABLED:
            return " [dim](思考模式: 已关闭)[/dim]"
        if self is ThinkingCapability.FEATURE_OFF:
            return " [dim](思考模式: 关闭功能未启用)[/dim]"
        return (
            " [bold yellow]⚠️ 思考模式: 无法关闭"
            "（使用默认强度）[/bold yellow]"
        )


@dataclass(frozen=True)
class ThinkingDisableSpec:
    """单个模型的关闭思考说明。"""

    method: ThinkingDisableMethod
    reasoning_effort: Optional[str] = None


@dataclass(frozen=True)
class ThinkingPlugin:
    """一条可插拔的思考适配规则。"""

    names: frozenset[str] = field(default_factory=frozenset)
    prefixes: tuple[str, ...] = ()
    providers: frozenset[str] = field(default_factory=frozenset)
    method: Optional[ThinkingDisableMethod] = None
    reasoning_effort: Optional[str] = None
    capability: ThinkingCapability = ThinkingCapability.DISABLED
    overrides_provider: bool = False
    match: Optional[Callable[[str], bool]] = None

    def match_rank(self, name: str) -> Optional[tuple[int, int]]:
        """越具体的匹配越小。不匹配则返回 None。"""
        if name in self.names:
            return (0, -len(name))
        matching = [prefix for prefix in self.prefixes if name.startswith(prefix)]
        if matching:
            return (1, -max(len(prefix) for prefix in matching))
        if self.match is not None and self.match(name):
            return (2, 0)
        return None


@dataclass(frozen=True)
class ThinkingPlan:
    """核心代码只消费这份计划，不认识具体模型。"""

    capability: ThinkingCapability
    method: Optional[ThinkingDisableMethod] = None
    reasoning_effort: Optional[str] = None
    log_state: str = "default"


@dataclass(frozen=True)
class DetectedReasoningNotice:
    """运行时发现仍在思考时的提示。"""

    level: str
    message: str
