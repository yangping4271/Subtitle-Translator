from ..types import ThinkingCapability, ThinkingDisableMethod, ThinkingPlugin

PLUGINS = (
    ThinkingPlugin(
        names=frozenset({"deepseek-v4-flash", "deepseek-v4-pro"}),
        prefixes=("deepseek-v4-flash-", "deepseek-v4-pro-"),
        method=ThinkingDisableMethod.THINKING_TYPE_DISABLED,
        capability=ThinkingCapability.DISABLED,
    ),
    ThinkingPlugin(
        names=frozenset({"kimi-k2.6", "kimi-k2.5"}),
        method=ThinkingDisableMethod.THINKING_TYPE_DISABLED,
        capability=ThinkingCapability.DISABLED,
    ),
    ThinkingPlugin(
        names=frozenset({"minimax-m3"}),
        method=ThinkingDisableMethod.THINKING_TYPE_DISABLED,
        capability=ThinkingCapability.DISABLED,
    ),
    ThinkingPlugin(
        names=frozenset(
            {
                "claude-sonnet-5",
                "claude-opus-5",
                "claude-haiku-4-5",
                "claude-sonnet-4-6",
                "claude-opus-4-6",
                "claude-sonnet-4-5",
                "claude-opus-4-5",
                "claude-opus-4-8",
                "claude-opus-4-7",
            }
        ),
        method=ThinkingDisableMethod.THINKING_TYPE_DISABLED,
        capability=ThinkingCapability.DISABLED,
    ),
    ThinkingPlugin(
        names=frozenset(
            {"doubao-seed-1-6", "doubao-seed-1-8", "doubao-seed-2-0"}
        ),
        method=ThinkingDisableMethod.THINKING_TYPE_DISABLED,
        capability=ThinkingCapability.DISABLED,
    ),
)
