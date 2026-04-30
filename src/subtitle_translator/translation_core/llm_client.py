"""LLM 客户端统一封装

提供 OpenAI API 调用的单一入口，便于后续扩展和维护。
"""
from typing import Optional
from openai import OpenAI

from ..logger import setup_logger
from .config import SubtitleConfig

logger = setup_logger("llm_client")


class LLMClient:
    """LLM 客户端封装类

    使用单例模式管理 OpenAI 客户端实例，避免重复创建连接。
    """

    _instance: Optional["LLMClient"] = None

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

    @classmethod
    def get_instance(cls, config: Optional[SubtitleConfig] = None) -> "LLMClient":
        """获取 LLM 客户端单例

        Args:
            config: 可选的配置对象，首次调用时必须提供

        Returns:
            LLMClient 实例
        """
        if cls._instance is None:
            if config is None:
                config = SubtitleConfig()
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置单例实例

        用于测试或需要重新初始化客户端的场景。
        """
        cls._instance = None

    @property
    def client(self) -> OpenAI:
        """获取底层 OpenAI 客户端

        提供对原始客户端的访问，保持与现有代码的兼容性。

        Returns:
            OpenAI 客户端实例
        """
        return self._client

    def _get_openai_reasoning_effort(self, model: str) -> Optional[str]:
        """OpenAI reasoning 模型在翻译任务中尽量关闭或降低推理。"""
        if not self.config.disable_thinking or self._provider_type != "openai":
            return None

        normalized_model = model.lower()
        if normalized_model.startswith(("gpt-5.1", "gpt-5.2")):
            return "none"
        if normalized_model.startswith("gpt-5") and "pro" not in normalized_model:
            return "minimal"
        return None

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

        if self._provider_type == "openrouter" and "reasoning" not in extra_body:
            extra_body["reasoning"] = {"effort": "none"}

        return extra_body

    def _apply_reasoning_options(self, kwargs: dict) -> dict:
        """为已知供应商追加关闭思考参数。"""
        request = dict(kwargs)
        model = str(request.get("model") or "")

        extra_body = self._build_extra_body(request)
        if extra_body:
            request["extra_body"] = extra_body

        reasoning_effort = self._get_openai_reasoning_effort(model)
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
