"""关闭思考模式的统一注册表。

两条路径：
- 官方供应商网址能识别、且该供应商有统一关闭参数时，按供应商关闭
- 其他端点只按已登记的模型名关闭

匹配模型名时会去掉 `openai/`、`deepseek/` 这类前缀，且大小写不敏感。
未登记模型在非供应商级端点上不会附加关闭参数。
请求路径和 console 展示共用本模块。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ThinkingDisableMethod(str, Enum):
    """已知可用的关闭思考方式。"""

    OPENAI_REASONING_EFFORT = "openai_reasoning_effort"
    THINKING_TYPE_DISABLED = "thinking_type_disabled"
    OPENROUTER_REASONING = "openrouter_reasoning"
    GOOGLE_THINKING_LEVEL = "google_thinking_level"
    GOOGLE_THINKING_BUDGET = "google_thinking_budget"
    ENABLE_THINKING_FALSE = "enable_thinking_false"


@dataclass(frozen=True)
class ThinkingDisableSpec:
    """单个模型的关闭思考说明。"""

    method: ThinkingDisableMethod
    reasoning_effort: Optional[str] = None


# 官方网址可识别、且有统一关闭参数的供应商。
# OpenAI / Kimi / Google / Groq / Anthropic / xAI 没有对所有模型都安全的关闭开关。
PROVIDER_THINKING_DISABLE: dict[str, ThinkingDisableMethod] = {
    "openrouter": ThinkingDisableMethod.OPENROUTER_REASONING,
    "deepseek": ThinkingDisableMethod.THINKING_TYPE_DISABLED,
    "zhipu": ThinkingDisableMethod.THINKING_TYPE_DISABLED,
    "minimax": ThinkingDisableMethod.THINKING_TYPE_DISABLED,
    "dashscope": ThinkingDisableMethod.ENABLE_THINKING_FALSE,
    "volcengine": ThinkingDisableMethod.THINKING_TYPE_DISABLED,
}


def normalize_model_name(model: str) -> str:
    """去掉 vendor 前缀，便于按官方模型名登记。"""
    return (model or "").strip().lower().rsplit("/", 1)[-1]


def _openai_none() -> ThinkingDisableSpec:
    return ThinkingDisableSpec(
        ThinkingDisableMethod.OPENAI_REASONING_EFFORT,
        reasoning_effort="none",
    )


def _thinking_disabled() -> ThinkingDisableSpec:
    return ThinkingDisableSpec(ThinkingDisableMethod.THINKING_TYPE_DISABLED)


def _google_thinking_level() -> ThinkingDisableSpec:
    return ThinkingDisableSpec(ThinkingDisableMethod.GOOGLE_THINKING_LEVEL)


def _google_thinking_budget() -> ThinkingDisableSpec:
    return ThinkingDisableSpec(ThinkingDisableMethod.GOOGLE_THINKING_BUDGET)


def _enable_thinking_false() -> ThinkingDisableSpec:
    return ThinkingDisableSpec(ThinkingDisableMethod.ENABLE_THINKING_FALSE)


# 在这里追加新模型即可。
THINKING_DISABLE_MODELS: dict[str, ThinkingDisableSpec] = {
    # OpenAI GPT-5.6
    "gpt-5.6": _openai_none(),
    "gpt-5.6-sol": _openai_none(),
    "gpt-5.6-terra": _openai_none(),
    "gpt-5.6-luna": _openai_none(),
    # DeepSeek V4
    "deepseek-v4-flash": _thinking_disabled(),
    "deepseek-v4-pro": _thinking_disabled(),
    # 智谱 GLM 主流思考模型
    "glm-5.2": _thinking_disabled(),
    "glm-5.1": _thinking_disabled(),
    "glm-5": _thinking_disabled(),
    "glm-5-turbo": _thinking_disabled(),
    "glm-4.7": _thinking_disabled(),
    "glm-4.6": _thinking_disabled(),
    "glm-4.5": _thinking_disabled(),
    # Kimi：K3 / K2.7-code 官方不允许关闭思考
    "kimi-k2.6": _thinking_disabled(),
    "kimi-k2.5": _thinking_disabled(),
    # MiniMax：官方仅 M3 能关；M2.x 传 disabled 仍会思考
    "minimax-m3": _thinking_disabled(),
    # Google Gemini 主流 Flash（OpenAI 兼容端点）
    # Gemini 3 用 thinking_level；Gemini 2.5 用 thinking_budget；不能同时发送。
    "gemini-3.6-flash": _google_thinking_level(),
    "gemini-3.5-flash": _google_thinking_level(),
    "gemini-3-flash-preview": _google_thinking_level(),
    "gemini-2.5-flash": _google_thinking_budget(),
    # Anthropic：Fable / Mythos 官方不允许关闭思考
    "claude-sonnet-5": _thinking_disabled(),
    "claude-opus-5": _thinking_disabled(),
    "claude-haiku-4-5": _thinking_disabled(),
    "claude-sonnet-4-6": _thinking_disabled(),
    "claude-opus-4-6": _thinking_disabled(),
    "claude-sonnet-4-5": _thinking_disabled(),
    "claude-opus-4-5": _thinking_disabled(),
    "claude-opus-4-8": _thinking_disabled(),
    "claude-opus-4-7": _thinking_disabled(),
    # xAI：grok-4.5 / grok-4.6 官方不能关思考
    "grok-4.3": _openai_none(),
    "grok-4.3-latest": _openai_none(),
    # 千问主流混合思考模型
    "qwen-plus": _enable_thinking_false(),
    "qwen-turbo": _enable_thinking_false(),
    "qwen-flash": _enable_thinking_false(),
    "qwen-max": _enable_thinking_false(),
    "qwen3-max": _enable_thinking_false(),
    "qwen3.5-plus": _enable_thinking_false(),
    "qwen3.5-flash": _enable_thinking_false(),
    "qwen3.6-plus": _enable_thinking_false(),
    "qwen3.6-flash": _enable_thinking_false(),
    "qwen3.7-plus": _enable_thinking_false(),
    "qwen3.7-max": _enable_thinking_false(),
    # 豆包主流 Seed
    "doubao-seed-1-6": _thinking_disabled(),
    "doubao-seed-1-8": _thinking_disabled(),
    "doubao-seed-2-0": _thinking_disabled(),
}


def get_thinking_disable_spec(model: str) -> Optional[ThinkingDisableSpec]:
    """查找模型的关闭思考配置；未登记则返回 None。"""
    return THINKING_DISABLE_MODELS.get(normalize_model_name(model))


def get_provider_thinking_method(
    provider_type: Optional[str],
) -> Optional[ThinkingDisableMethod]:
    """返回供应商级关闭方式；该供应商没有统一开关则返回 None。"""
    if not provider_type:
        return None
    return PROVIDER_THINKING_DISABLE.get(provider_type)


def thinking_disable_applies(model: str, provider_type: Optional[str] = None) -> bool:
    """请求路径和 console 共用：供应商级开关或登记模型才关闭思考。"""
    if get_provider_thinking_method(provider_type) is not None:
        return True
    return get_thinking_disable_spec(model) is not None


def encode_thinking_extra_body(
    extra_body: dict,
    method: ThinkingDisableMethod,
) -> dict:
    """把关闭方式写进 extra_body。"""
    encoded = dict(extra_body)
    if method is ThinkingDisableMethod.OPENROUTER_REASONING:
        encoded["reasoning"] = {"effort": "none"}
    elif method is ThinkingDisableMethod.THINKING_TYPE_DISABLED:
        encoded["thinking"] = {"type": "disabled"}
    elif method is ThinkingDisableMethod.GOOGLE_THINKING_LEVEL:
        encoded["extra_body"] = {
            "google": {
                "thinking_config": {
                    "thinking_level": "minimal",
                }
            }
        }
    elif method is ThinkingDisableMethod.GOOGLE_THINKING_BUDGET:
        encoded["extra_body"] = {
            "google": {
                "thinking_config": {
                    "thinking_budget": 0,
                }
            }
        }
    elif method is ThinkingDisableMethod.ENABLE_THINKING_FALSE:
        encoded["enable_thinking"] = False
    return encoded
