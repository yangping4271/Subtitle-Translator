from ..types import ThinkingCapability, ThinkingPlugin

# 官方不能关闭思考：不发关闭参数，也不降档，沿用默认强度。
# overrides_provider 避免供应商级开关把 thinking.disabled 发给这些模型。
PLUGINS = (
    ThinkingPlugin(
        names=frozenset({"glm-5.3"}),
        prefixes=("glm-5.3-",),
        capability=ThinkingCapability.DEFAULT,
        overrides_provider=True,
    ),
    ThinkingPlugin(
        prefixes=("grok-4.5", "grok-4.6"),
        capability=ThinkingCapability.DEFAULT,
        overrides_provider=True,
    ),
)
