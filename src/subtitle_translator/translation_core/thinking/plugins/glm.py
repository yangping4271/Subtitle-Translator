from ..types import ThinkingCapability, ThinkingDisableMethod, ThinkingPlugin

PLUGINS = (
    ThinkingPlugin(
        names=frozenset({"glm-5.3"}),
        prefixes=("glm-5.3-",),
        method=ThinkingDisableMethod.OPENAI_REASONING_EFFORT,
        reasoning_effort="low",
        capability=ThinkingCapability.MIN_REASONING,
        overrides_provider=True,
    ),
    ThinkingPlugin(
        names=frozenset(
            {
                "glm-5.2",
                "glm-5.1",
                "glm-5",
                "glm-5-turbo",
                "glm-4.7",
                "glm-4.6",
                "glm-4.5",
            }
        ),
        method=ThinkingDisableMethod.THINKING_TYPE_DISABLED,
        capability=ThinkingCapability.DISABLED,
    ),
)
