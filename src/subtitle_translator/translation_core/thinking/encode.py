"""把思考策略编码成请求字段。"""

from .types import ThinkingDisableMethod, ThinkingPlan


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


def log_state_for_plan(plan: ThinkingPlan) -> str:
    """请求日志用的思考状态标签。"""
    method = plan.method
    if method is ThinkingDisableMethod.THINKING_TYPE_DISABLED:
        return "thinking-disabled"
    if method is ThinkingDisableMethod.OPENROUTER_REASONING:
        return "openrouter-none"
    if method is ThinkingDisableMethod.ENABLE_THINKING_FALSE:
        return "enable-thinking-false"
    if method is ThinkingDisableMethod.GOOGLE_THINKING_LEVEL:
        return "google-minimal"
    if method is ThinkingDisableMethod.GOOGLE_THINKING_BUDGET:
        return "google-budget-0"
    if method is ThinkingDisableMethod.OPENAI_REASONING_EFFORT and plan.reasoning_effort:
        return f"openai-{plan.reasoning_effort}"
    return "default"
