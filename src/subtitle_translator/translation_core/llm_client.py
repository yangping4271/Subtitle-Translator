"""LLM 客户端统一封装

提供 OpenAI API 调用的单一入口，便于后续扩展和维护。
"""
import re
import time
from typing import Any, Optional, Protocol
from openai import OpenAI

from ..logger import setup_logger
from .config import SubtitleConfig, validate_api_configuration

logger = setup_logger("llm_client")


def _field(value: Any, name: str) -> Any:
    """兼容 SDK 对象和第三方端点返回的字典。"""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _integer_metric(value: Any) -> Optional[int]:
    """仅保留真实的整数用量，避免 Mock/缺失字段污染日志。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def get_response_usage(response: Any) -> dict[str, Optional[int | str]]:
    """提取常见 Chat Completions 用量字段。"""
    usage = _field(response, "usage")
    completion_details = _field(usage, "completion_tokens_details")
    response_id = _field(response, "id")
    return {
        "request_id": response_id if isinstance(response_id, str) else None,
        "prompt_tokens": _integer_metric(
            _field(usage, "prompt_tokens") or _field(usage, "input_tokens")
        ),
        "completion_tokens": _integer_metric(
            _field(usage, "completion_tokens") or _field(usage, "output_tokens")
        ),
        "total_tokens": _integer_metric(_field(usage, "total_tokens")),
        "reasoning_tokens": _integer_metric(
            _field(completion_details, "reasoning_tokens")
        ),
    }


def _request_text_size(messages: Any) -> tuple[int, int]:
    """计算提示消息的字符数和粗略词数，不记录实际内容。"""
    if not isinstance(messages, list):
        return 0, 0
    contents = []
    for message in messages:
        content = _field(message, "content")
        if isinstance(content, str):
            contents.append(content)
    text = "\n".join(contents)
    return len(text), len(re.findall(r"\S+", text))


class ModelAdapter(Protocol):
    """TranslationEngine 和 SubtitleSegmenter 使用的 external seam。"""

    def create_chat_completion(self, **kwargs: Any) -> Any:
        """创建一次聊天补全请求。"""
        ...


def get_reasoning_effort(model: str) -> Optional[str]:
    """返回字幕任务可用的最低推理强度；优先彻底关闭推理。"""
    normalized_model = model.lower().rsplit("/", 1)[-1]
    if re.search(r"(?:^|-)pro(?:-|$)", normalized_model):
        return None

    version_match = re.match(r"^gpt-(\d+)(?:\.(\d+))?(?:-|$)", normalized_model)
    if not version_match:
        return None

    major = int(version_match.group(1))
    minor = int(version_match.group(2) or 0)
    if major > 5 or (major == 5 and minor >= 1):
        return "none"
    if major == 5:
        # 原始 GPT-5 系列不支持 none，minimal 是其官方最低档。
        return "minimal"
    return None


class LLMClient:
    """一次运行所有的 OpenAI-compatible adapter。"""

    def __init__(self, config: SubtitleConfig):
        """初始化 LLM 客户端

        Args:
            config: 字幕翻译配置对象
        """
        validate_api_configuration(config.openai_base_url, config.openai_api_key)
        self.config = config
        self._client = OpenAI(
            base_url=config.openai_base_url,
            api_key=config.openai_api_key
        )
        self._provider_type = config.provider_type()

    def _get_reasoning_effort(self, model: str) -> Optional[str]:
        """获取模型支持的关闭或最低推理强度。"""
        return get_reasoning_effort(model)

    def _build_extra_body(self, kwargs: dict) -> dict:
        """构建供应商扩展参数，并强制执行禁用思考策略。"""
        extra_body = dict(kwargs.get("extra_body") or {})
        if not self.config.disable_thinking:
            return extra_body

        model = str(kwargs.get("model") or "")
        normalized_model = model.lower()

        if (
            self._provider_type == "deepseek"
            and normalized_model.startswith("deepseek-v4-")
        ):
            extra_body["thinking"] = {"type": "disabled"}

        if self._provider_type == "openrouter":
            extra_body["reasoning"] = {"effort": "none"}

        if self._provider_type == "dashscope":
            if normalized_model.startswith("minimax"):
                extra_body["thinking"] = {"type": "disabled"}
            else:
                extra_body["enable_thinking"] = False

        return extra_body

    def _apply_reasoning_options(self, kwargs: dict) -> dict:
        """为已知供应商追加关闭思考参数。"""
        request = dict(kwargs)
        model = str(request.get("model") or "")

        extra_body = self._build_extra_body(request)
        if extra_body:
            request["extra_body"] = extra_body

        reasoning_effort = None
        if self.config.disable_thinking:
            # OpenRouter 和 DashScope 使用各自的扩展参数；其他
            # OpenAI-compatible 端点沿用 Chat Completions 的 reasoning_effort。
            request.pop("reasoning_effort", None)
            reasoning_effort = self._get_reasoning_effort(model)
        if reasoning_effort and self._provider_type not in {"openrouter", "dashscope"}:
            request["reasoning_effort"] = reasoning_effort

        reasoning_state = "default"
        if extra_body.get("thinking") == {"type": "disabled"}:
            reasoning_state = "thinking-disabled"
        elif extra_body.get("reasoning") == {"effort": "none"}:
            reasoning_state = "openrouter-none"
        elif extra_body.get("enable_thinking") is False:
            reasoning_state = "dashscope-disabled"
        elif reasoning_effort and self._provider_type not in {"openrouter", "dashscope"}:
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
        input_chars, input_words = _request_text_size(request.get("messages"))
        start_time = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
        except Exception:
            logger.warning(
                "请求失败: model=%s, latency=%.2fs, input_chars=%s, input_words=%s",
                request.get("model", ""),
                time.perf_counter() - start_time,
                input_chars,
                input_words,
            )
            raise

        elapsed = time.perf_counter() - start_time
        usage = get_response_usage(response)
        logger.info(
            "请求完成: model=%s, request_id=%s, latency=%.2fs, "
            "input_chars=%s, input_words=%s, prompt_tokens=%s, "
            "completion_tokens=%s, total_tokens=%s, reasoning_tokens=%s",
            request.get("model", ""),
            usage["request_id"] or "unknown",
            elapsed,
            input_chars,
            input_words,
            usage["prompt_tokens"]
            if usage["prompt_tokens"] is not None
            else "unknown",
            usage["completion_tokens"]
            if usage["completion_tokens"] is not None
            else "unknown",
            usage["total_tokens"]
            if usage["total_tokens"] is not None
            else "unknown",
            usage["reasoning_tokens"]
            if usage["reasoning_tokens"] is not None
            else "unknown",
        )
        return response

    def close(self) -> None:
        """释放当前 adapter 拥有的底层连接。"""
        self._client.close()
