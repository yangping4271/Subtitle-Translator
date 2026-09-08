"""关闭思考模式的插件入口。

新增模型：在 ``plugins/`` 下加模块并导出 ``PLUGINS``。
核心请求路径只调用 ``apply_thinking_options`` / ``resolve_thinking_plan``。
"""

from .catalog import (
    apply_thinking_options,
    detected_reasoning_notice,
    get_provider_thinking_method,
    get_reasoning_effort,
    get_thinking_disable_spec,
    normalize_model_name,
    resolve_thinking_plan,
    thinking_cannot_disable,
    thinking_console_suffix,
    thinking_disable_applies,
    thinking_uses_min_reasoning,
)
from .encode import encode_thinking_extra_body
from .plugin import register_plugin, unregister_plugin
from .types import (
    DetectedReasoningNotice,
    ThinkingCapability,
    ThinkingDisableMethod,
    ThinkingDisableSpec,
    ThinkingPlan,
    ThinkingPlugin,
)

__all__ = [
    "DetectedReasoningNotice",
    "ThinkingCapability",
    "ThinkingDisableMethod",
    "ThinkingDisableSpec",
    "ThinkingPlan",
    "ThinkingPlugin",
    "apply_thinking_options",
    "detected_reasoning_notice",
    "encode_thinking_extra_body",
    "get_provider_thinking_method",
    "get_reasoning_effort",
    "get_thinking_disable_spec",
    "normalize_model_name",
    "register_plugin",
    "resolve_thinking_plan",
    "thinking_cannot_disable",
    "thinking_console_suffix",
    "thinking_disable_applies",
    "thinking_uses_min_reasoning",
    "unregister_plugin",
]
