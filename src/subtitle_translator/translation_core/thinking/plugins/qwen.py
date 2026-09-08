from ..types import ThinkingCapability, ThinkingDisableMethod, ThinkingPlugin

PLUGINS = (
    ThinkingPlugin(
        names=frozenset(
            {
                "qwen-plus",
                "qwen-turbo",
                "qwen-flash",
                "qwen-max",
                "qwen3-max",
                "qwen3.5-plus",
                "qwen3.5-flash",
                "qwen3.6-plus",
                "qwen3.6-flash",
                "qwen3.7-plus",
                "qwen3.7-max",
            }
        ),
        method=ThinkingDisableMethod.ENABLE_THINKING_FALSE,
        capability=ThinkingCapability.DISABLED,
    ),
    ThinkingPlugin(
        match=lambda name: name.startswith("qwen3"),
        method=ThinkingDisableMethod.ENABLE_THINKING_FALSE,
        capability=ThinkingCapability.DISABLED,
    ),
)
