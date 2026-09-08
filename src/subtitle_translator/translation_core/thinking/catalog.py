"""根据插件解析思考计划；核心请求路径只调用这里。"""

from typing import Optional

from .encode import encode_thinking_extra_body, log_state_for_plan
from .plugin import iter_plugins
from .types import (
    DetectedReasoningNotice,
    ThinkingCapability,
    ThinkingDisableMethod,
    ThinkingDisableSpec,
    ThinkingPlan,
    ThinkingPlugin,
)


def normalize_model_name(model: str) -> str:
    """去掉 vendor 前缀，便于按官方模型名匹配。"""
    return (model or "").strip().lower().rsplit("/", 1)[-1]


def _is_provider_plugin(plugin: ThinkingPlugin) -> bool:
    return bool(plugin.providers) and not plugin.names and not plugin.prefixes and plugin.match is None


def _find_provider_plugin(provider_type: Optional[str]) -> Optional[ThinkingPlugin]:
    if not provider_type:
        return None
    for plugin in iter_plugins():
        if _is_provider_plugin(plugin) and provider_type in plugin.providers:
            return plugin
    return None


def _find_model_plugin(
    model: str,
    capability: Optional[ThinkingCapability] = None,
) -> Optional[ThinkingPlugin]:
    name = normalize_model_name(model)
    ranked: list[tuple[tuple[int, int], ThinkingPlugin]] = []
    for plugin in iter_plugins():
        if _is_provider_plugin(plugin):
            continue
        if capability is not None and plugin.capability is not capability:
            continue
        if capability is None and plugin.capability is ThinkingCapability.CANNOT_DISABLE:
            continue
        rank = plugin.match_rank(name)
        if rank is not None:
            ranked.append((rank, plugin))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def get_thinking_disable_spec(model: str) -> Optional[ThinkingDisableSpec]:
    """查找模型的关闭或降档配置；未登记则返回 None。"""
    plugin = _find_model_plugin(model)
    if plugin is None or plugin.method is None:
        return None
    return ThinkingDisableSpec(plugin.method, plugin.reasoning_effort)


def get_provider_thinking_method(
    provider_type: Optional[str],
) -> Optional[ThinkingDisableMethod]:
    """返回供应商级关闭方式；该供应商没有统一开关则返回 None。"""
    plugin = _find_provider_plugin(provider_type)
    if plugin is None:
        return None
    return plugin.method


def thinking_uses_min_reasoning(model: str) -> bool:
    """关闭思考时只会降到最低推理强度，而不是完全关闭。"""
    plugin = _find_model_plugin(model)
    return plugin is not None and plugin.capability is ThinkingCapability.MIN_REASONING


def thinking_cannot_disable(model: str) -> bool:
    """已知官方不允许关闭思考、且没有可用最低强度参数的模型。"""
    plugin = _find_model_plugin(model, ThinkingCapability.CANNOT_DISABLE)
    return plugin is not None


def thinking_disable_applies(model: str, provider_type: Optional[str] = None) -> bool:
    """请求路径和 console 共用：供应商级开关或登记模型才会改思考参数。"""
    plan = resolve_thinking_plan(model, provider_type, disable_thinking=True)
    return plan.capability in {
        ThinkingCapability.DISABLED,
        ThinkingCapability.MIN_REASONING,
    }


def get_reasoning_effort(model: str) -> Optional[str]:
    """返回已登记模型的 reasoning_effort；其他方式返回 None。"""
    spec = get_thinking_disable_spec(model)
    if spec is None:
        return None
    if spec.method is ThinkingDisableMethod.OPENAI_REASONING_EFFORT:
        return spec.reasoning_effort
    return None


def _plan_from_plugin(plugin: ThinkingPlugin) -> ThinkingPlan:
    plan = ThinkingPlan(
        capability=plugin.capability,
        method=plugin.method,
        reasoning_effort=plugin.reasoning_effort,
    )
    return ThinkingPlan(
        capability=plan.capability,
        method=plan.method,
        reasoning_effort=plan.reasoning_effort,
        log_state=log_state_for_plan(plan),
    )


def resolve_thinking_plan(
    model: str,
    provider_type: Optional[str] = None,
    disable_thinking: bool = True,
) -> ThinkingPlan:
    """解析这次请求该用的思考策略。"""
    if not disable_thinking:
        return ThinkingPlan(capability=ThinkingCapability.FEATURE_OFF)

    model_plugin = _find_model_plugin(model)
    provider_plugin = _find_provider_plugin(provider_type)
    if model_plugin is not None and model_plugin.overrides_provider:
        return _plan_from_plugin(model_plugin)
    if provider_plugin is not None:
        return _plan_from_plugin(provider_plugin)
    if model_plugin is not None:
        return _plan_from_plugin(model_plugin)
    if thinking_cannot_disable(model):
        return ThinkingPlan(capability=ThinkingCapability.CANNOT_DISABLE)
    return ThinkingPlan(capability=ThinkingCapability.UNADAPTED)


def apply_thinking_options(
    kwargs: dict,
    provider_type: Optional[str],
    disable_thinking: bool,
) -> tuple[dict, ThinkingPlan]:
    """按计划改写请求，不在调用方写模型分支。"""
    request = dict(kwargs)
    model = str(request.get("model") or "")
    plan = resolve_thinking_plan(model, provider_type, disable_thinking)
    extra_body = dict(request.get("extra_body") or {})
    if (
        plan.method is not None
        and plan.method is not ThinkingDisableMethod.OPENAI_REASONING_EFFORT
    ):
        extra_body = encode_thinking_extra_body(extra_body, plan.method)
    if extra_body:
        request["extra_body"] = extra_body
    if disable_thinking:
        request.pop("reasoning_effort", None)
        if plan.reasoning_effort:
            request["reasoning_effort"] = plan.reasoning_effort
    return request, plan


def thinking_console_suffix(
    model: str,
    provider_type: Optional[str] = None,
    disable_thinking: bool = True,
) -> str:
    """配置行里思考状态的展示文案。"""
    return resolve_thinking_plan(model, provider_type, disable_thinking).capability.console_suffix()


def detected_reasoning_notice(
    model: str,
    provider_type: Optional[str],
    disable_thinking: bool,
    evidence: str,
) -> DetectedReasoningNotice:
    """响应仍在思考时的提示；文案由能力决定。"""
    display_model = model or "unknown"
    plan = resolve_thinking_plan(display_model, provider_type, disable_thinking)
    capability = plan.capability
    if capability is ThinkingCapability.MIN_REASONING:
        return DetectedReasoningNotice(
            "info",
            f"该模型无法关闭思考，已使用最低强度: {display_model}, {evidence}",
        )
    if capability is ThinkingCapability.DISABLED:
        return DetectedReasoningNotice(
            "warning",
            "检测到模型实际使用了思考模式（关闭参数未生效）: "
            f"{display_model}, {evidence}",
        )
    if capability is ThinkingCapability.CANNOT_DISABLE:
        return DetectedReasoningNotice(
            "info",
            f"该模型无法关闭思考，使用默认强度: {display_model}, {evidence}",
        )
    if capability is ThinkingCapability.UNADAPTED:
        return DetectedReasoningNotice(
            "warning",
            f"未适配该模型的关闭方式，使用默认强度: {display_model}, {evidence}",
        )
    return DetectedReasoningNotice(
        "info",
        f"检测到模型实际使用了思考模式: {display_model}, {evidence}",
    )
