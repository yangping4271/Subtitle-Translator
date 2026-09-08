from ..types import ThinkingCapability, ThinkingDisableMethod, ThinkingPlugin

PLUGINS = (
    ThinkingPlugin(
        names=frozenset({"grok-4.3", "grok-4.3-latest"}),
        method=ThinkingDisableMethod.OPENAI_REASONING_EFFORT,
        reasoning_effort="none",
        capability=ThinkingCapability.DISABLED,
    ),
)
