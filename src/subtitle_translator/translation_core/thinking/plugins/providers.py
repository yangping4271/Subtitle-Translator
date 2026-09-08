from ..types import ThinkingCapability, ThinkingDisableMethod, ThinkingPlugin

PLUGINS = (
    ThinkingPlugin(
        providers=frozenset({"openrouter"}),
        method=ThinkingDisableMethod.OPENROUTER_REASONING,
        capability=ThinkingCapability.DISABLED,
    ),
    ThinkingPlugin(
        providers=frozenset({"deepseek", "zhipu", "minimax", "volcengine"}),
        method=ThinkingDisableMethod.THINKING_TYPE_DISABLED,
        capability=ThinkingCapability.DISABLED,
    ),
    ThinkingPlugin(
        providers=frozenset({"dashscope"}),
        method=ThinkingDisableMethod.ENABLE_THINKING_FALSE,
        capability=ThinkingCapability.DISABLED,
    ),
)
