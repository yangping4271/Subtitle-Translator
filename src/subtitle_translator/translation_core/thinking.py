"""关闭思考模式的统一注册表。

新增模型时只改 THINKING_DISABLE_MODELS：
- key 使用官方 API 模型名，不含 vendor 前缀
- value 写明该模型的关闭方式

匹配时会去掉 `openai/`、`deepseek/` 这类前缀，且大小写不敏感。
未登记模型不会因为名称相近而被附加关闭思考参数。
OpenRouter 仅在模型已登记时改用其供应商编码。
请求路径和 console 展示共用本模块。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ThinkingDisableMethod(str, Enum):
    """已知可用的关闭思考方式。"""

    OPENAI_REASONING_EFFORT = "openai_reasoning_effort"
    DEEPSEEK_THINKING = "deepseek_thinking"


@dataclass(frozen=True)
class ThinkingDisableSpec:
    """单个模型的关闭思考说明。"""

    method: ThinkingDisableMethod
    reasoning_effort: Optional[str] = None


def normalize_model_name(model: str) -> str:
    """去掉 vendor 前缀，便于按官方模型名登记。"""
    return (model or "").strip().lower().rsplit("/", 1)[-1]


# 在这里追加新模型即可。
THINKING_DISABLE_MODELS: dict[str, ThinkingDisableSpec] = {
    # GPT-5.6：Sol / Terra / Luna，以及指向 Sol 的官方别名。
    "gpt-5.6": ThinkingDisableSpec(
        ThinkingDisableMethod.OPENAI_REASONING_EFFORT,
        reasoning_effort="none",
    ),
    "gpt-5.6-sol": ThinkingDisableSpec(
        ThinkingDisableMethod.OPENAI_REASONING_EFFORT,
        reasoning_effort="none",
    ),
    "gpt-5.6-terra": ThinkingDisableSpec(
        ThinkingDisableMethod.OPENAI_REASONING_EFFORT,
        reasoning_effort="none",
    ),
    "gpt-5.6-luna": ThinkingDisableSpec(
        ThinkingDisableMethod.OPENAI_REASONING_EFFORT,
        reasoning_effort="none",
    ),
    # DeepSeek V4：官方默认开启思考。
    "deepseek-v4-flash": ThinkingDisableSpec(
        ThinkingDisableMethod.DEEPSEEK_THINKING,
    ),
    "deepseek-v4-pro": ThinkingDisableSpec(
        ThinkingDisableMethod.DEEPSEEK_THINKING,
    ),
}


def get_thinking_disable_spec(model: str) -> Optional[ThinkingDisableSpec]:
    """查找模型的关闭思考配置；未登记则返回 None。"""
    return THINKING_DISABLE_MODELS.get(normalize_model_name(model))


def thinking_disable_applies(model: str, provider_type: Optional[str] = None) -> bool:
    """请求路径和 console 共用：只有登记模型才关闭思考。"""
    del provider_type
    return get_thinking_disable_spec(model) is not None
