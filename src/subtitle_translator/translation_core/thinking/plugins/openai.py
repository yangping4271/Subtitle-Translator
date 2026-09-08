from ..types import ThinkingCapability, ThinkingDisableMethod, ThinkingPlugin

PLUGINS = (
    ThinkingPlugin(
        names=frozenset({"gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}),
        method=ThinkingDisableMethod.OPENAI_REASONING_EFFORT,
        reasoning_effort="none",
        capability=ThinkingCapability.DISABLED,
    ),
)
