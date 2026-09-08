from ..types import ThinkingCapability, ThinkingDisableMethod, ThinkingPlugin

PLUGINS = (
    ThinkingPlugin(
        names=frozenset({"grok-4.3", "grok-4.3-latest"}),
        method=ThinkingDisableMethod.OPENAI_REASONING_EFFORT,
        reasoning_effort="none",
        capability=ThinkingCapability.DISABLED,
    ),
    ThinkingPlugin(
        prefixes=("grok-4.5", "grok-4.6"),
        method=ThinkingDisableMethod.OPENAI_REASONING_EFFORT,
        reasoning_effort="low",
        capability=ThinkingCapability.MIN_REASONING,
        overrides_provider=True,
    ),
)
