from ..types import ThinkingCapability, ThinkingPlugin

PLUGINS = (
    ThinkingPlugin(
        names=frozenset(
            {
                "kimi-k3",
                "kimi-k2.7-code",
                "minimax-m2.7",
                "claude-fable-5",
                "claude-mythos-5",
                "gemini-2.5-pro",
            }
        ),
        capability=ThinkingCapability.CANNOT_DISABLE,
    ),
)
