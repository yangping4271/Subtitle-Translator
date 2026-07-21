"""LLM 客户端统一封装

提供 OpenAI API 调用的单一入口，便于后续扩展和维护。
"""
from typing import Any, Optional, Protocol
from openai import OpenAI

from ..logger import setup_logger
from .config import SubtitleConfig

logger = setup_logger("llm_client")


class ModelAdapter(Protocol):
    """TranslationEngine 和 SubtitleSegmenter 使用的 external seam。"""

    def create_chat_completion(self, **kwargs: Any) -> Any:
        """创建一次聊天补全请求。"""
        ...


def get_reasoning_effort(model: str) -> Optional[str]:
    """返回模型在字幕任务中使用的推理强度。"""
    normalized_model = model.lower().rsplit("/", 1)[-1]
    model_version = (
        normalized_model.removeprefix("gpt-")
        .split(".", 1)[0]
        .split("-", 1)[0]
    )
    if (
        normalized_model.startswith("gpt-")
        and model_version.isdigit()
        and int(model_version) >= 5
        and "pro" not in normalized_model
    ):
        return "low"
    return None


class LLMClient:
    """一次运行所有的 OpenAI-compatible adapter。"""

    def __init__(self, config: SubtitleConfig):
        """初始化 LLM 客户端

        Args:
            config: 字幕翻译配置对象
        """
        self.config = config
        self._client = OpenAI(
            base_url=config.openai_base_url,
            api_key=config.openai_api_key
        )
        self._provider_type = config.provider_type()

    def _get_reasoning_effort(self, model: str) -> Optional[str]:
        """GPT-5 及后续主版本在字幕任务中统一使用低推理强度。"""
        return get_reasoning_effort(model)

    def _build_extra_body(self, kwargs: dict) -> dict:
        """构建供应商扩展参数，不覆盖调用方显式传入的 extra_body。"""
        extra_body = dict(kwargs.get("extra_body") or {})
        if not self.config.disable_thinking:
            return extra_body

        model = str(kwargs.get("model") or "")
        normalized_model = model.lower()

        if (
            self._provider_type == "deepseek"
            and normalized_model.startswith("deepseek-v4-")
            and "thinking" not in extra_body
        ):
            extra_body["thinking"] = {"type": "disabled"}

        if (
            self._provider_type == "openrouter"
            and not self._get_reasoning_effort(model)
            and "reasoning" not in extra_body
        ):
            extra_body["reasoning"] = {"effort": "none"}

        return extra_body

    def _apply_reasoning_options(self, kwargs: dict) -> dict:
        """为已知供应商追加关闭思考参数。"""
        request = dict(kwargs)
        model = str(request.get("model") or "")

        extra_body = self._build_extra_body(request)
        if extra_body:
            request["extra_body"] = extra_body

        reasoning_effort = self._get_reasoning_effort(model)
        if reasoning_effort and "reasoning_effort" not in request:
            request["reasoning_effort"] = reasoning_effort

        reasoning_state = "default"
        if extra_body.get("thinking") == {"type": "disabled"}:
            reasoning_state = "deepseek-disabled"
        elif extra_body.get("reasoning") == {"effort": "none"}:
            reasoning_state = "openrouter-none"
        elif reasoning_effort:
            reasoning_state = f"openai-{reasoning_effort}"

        response_format = request.get("response_format")
        response_format_type = (
            response_format.get("type")
            if isinstance(response_format, dict)
            else "none"
        )
        logger.info(
            "请求参数: provider=%s, model=%s, response_format=%s, reasoning=%s",
            self._provider_type,
            model,
            response_format_type,
            reasoning_state,
        )

        return request

    def create_chat_completion(self, **kwargs):
        """统一创建聊天补全请求。"""
        request = self._apply_reasoning_options(kwargs)
        return self._client.chat.completions.create(**request)

    def close(self) -> None:
        """释放当前 adapter 拥有的底层连接。"""
        self._client.close()
