"""LLM 客户端统一封装

提供 OpenAI API 调用的单一入口，便于后续扩展和维护。
"""

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol
from openai import OpenAI
from rich import print as rich_print
from rich.markup import escape

from ..logger import setup_logger
from .config import SubtitleConfig, validate_api_configuration
from .thinking import apply_thinking_options, detected_reasoning_notice

logger = setup_logger("llm_client")

SLOW_REQUEST_SECONDS = 30.0
CONTEXT_REFERENCE_TOKENS = 4096


@dataclass(frozen=True)
class RequestMetric:
    """一次非流式模型请求的性能与响应状态。"""

    started_at: float
    ended_at: float
    latency: float
    success: bool
    completion_tokens: Optional[int]
    finish_reason: Optional[str]
    content_chars: int
    error_type: Optional[str]
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


def _average(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    """使用线性插值计算分位数。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _interval_union_duration(intervals: list[tuple[float, float]]) -> float:
    """计算请求忙碌时间；并发重叠区间只累计一次。"""
    if not intervals:
        return 0.0

    ordered = sorted(intervals)
    current_start, current_end = ordered[0]
    duration = 0.0
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        duration += max(0.0, current_end - current_start)
        current_start, current_end = start, end
    return duration + max(0.0, current_end - current_start)


def _field(value: Any, name: str) -> Any:
    """兼容 SDK 对象和第三方端点返回的字典。"""
    try:
        if isinstance(value, dict):
            return value.get(name)
        return getattr(value, name, None)
    except Exception:
        # 响应统计属于观测能力，第三方对象的异常属性不能影响正常结果。
        return None


def _integer_metric(value: Any) -> Optional[int]:
    """仅保留真实的整数用量，避免 Mock/缺失字段污染日志。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _first_integer_metric(*values: Any) -> Optional[int]:
    """返回第一个有效的非负整数指标。"""
    for value in values:
        metric = _integer_metric(value)
        if metric is not None:
            return metric
    return None


def _maximum_integer_metric(*values: Any) -> Optional[int]:
    """兼容多个字段位置，并优先采用能证明实际用量的最大值。"""
    metrics = [
        metric for value in values if (metric := _integer_metric(value)) is not None
    ]
    return max(metrics) if metrics else None


def get_response_usage(response: Any) -> dict[str, Optional[int | str]]:
    """提取常见 Chat Completions 用量字段。"""
    usage = _field(response, "usage")
    completion_details = _field(usage, "completion_tokens_details")
    output_details = _field(usage, "output_tokens_details")
    response_id = _field(response, "id")
    return {
        "request_id": response_id if isinstance(response_id, str) else None,
        "prompt_tokens": _first_integer_metric(
            _field(usage, "prompt_tokens"),
            _field(usage, "input_tokens"),
        ),
        "completion_tokens": _first_integer_metric(
            _field(usage, "completion_tokens"),
            _field(usage, "output_tokens"),
        ),
        "total_tokens": _integer_metric(_field(usage, "total_tokens")),
        "reasoning_tokens": _maximum_integer_metric(
            _field(completion_details, "reasoning_tokens"),
            _field(output_details, "reasoning_tokens"),
            _field(usage, "reasoning_tokens"),
        ),
    }


def _response_finish_reason(response: Any) -> Optional[str]:
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return None
    finish_reason = _field(choices[0], "finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def _response_content_size(response: Any) -> int:
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return 0
    content = _field(_field(choices[0], "message"), "content")
    return len(content) if isinstance(content, str) else 0


def _response_has_reasoning_content(response: Any) -> bool:
    """兼容 DeepSeek 等通过消息字段返回思考内容的端点。"""
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return False
    reasoning_content = _field(
        _field(choices[0], "message"),
        "reasoning_content",
    )
    return isinstance(reasoning_content, str) and bool(reasoning_content.strip())


def _context_tokens(metric: RequestMetric) -> Optional[int]:
    """一次请求占用的上下文长度：优先 total，否则 prompt+completion。"""
    if metric.total_tokens is not None:
        return metric.total_tokens
    if metric.prompt_tokens is None:
        return None
    return metric.prompt_tokens + (metric.completion_tokens or 0)


def summarize_request_metrics(metrics: list[RequestMetric]) -> dict[str, Any]:
    """汇总一次字幕文件处理期间的所有模型请求。"""
    successful = [metric for metric in metrics if metric.success]
    latencies = [metric.latency for metric in metrics]
    known_token_metrics = [
        metric for metric in successful if metric.completion_tokens is not None
    ]
    token_busy_time = _interval_union_duration(
        [(metric.started_at, metric.ended_at) for metric in known_token_metrics]
    )
    missing_usage = len(successful) - len(known_token_metrics)
    empty_responses = sum(1 for metric in successful if metric.content_chars == 0)
    abnormal_finishes = sum(
        1 for metric in successful if metric.finish_reason not in {None, "stop"}
    )
    unknown_finishes = sum(1 for metric in successful if metric.finish_reason is None)
    error_types: dict[str, int] = {}
    for metric in metrics:
        if not metric.error_type:
            continue
        error_types[metric.error_type] = error_types.get(metric.error_type, 0) + 1

    anomalous_responses = sum(
        1
        for metric in successful
        if (
            metric.content_chars == 0
            or metric.finish_reason not in {None, "stop"}
            or metric.finish_reason is None
            or metric.completion_tokens is None
        )
    )
    context_sizes = [
        size
        for metric in successful
        if (size := _context_tokens(metric)) is not None
    ]
    max_context_tokens = max(context_sizes) if context_sizes else None

    return {
        "requests": len(metrics),
        "successful_requests": len(successful),
        "failed_requests": len(metrics) - len(successful),
        "latency_avg": _average(latencies),
        "latency_p95": _percentile(latencies, 0.95),
        "latency_max": max(latencies) if latencies else None,
        "effective_tps": (
            sum(metric.completion_tokens or 0 for metric in known_token_metrics)
            / token_busy_time
            if known_token_metrics and token_busy_time > 0
            else None
        ),
        "throughput_requests": len(known_token_metrics),
        "slow_requests": sum(
            1 for metric in metrics if metric.latency > SLOW_REQUEST_SECONDS
        ),
        "slowest_request": max((metric.latency for metric in metrics), default=None),
        "anomalies": anomalous_responses,
        "empty_responses": empty_responses,
        "abnormal_finishes": abnormal_finishes,
        "unknown_finishes": unknown_finishes,
        "missing_usage": missing_usage,
        "error_types": error_types,
        "max_context_tokens": max_context_tokens,
        "context_reference_tokens": CONTEXT_REFERENCE_TOKENS,
        "max_context_ratio": (
            max_context_tokens / CONTEXT_REFERENCE_TOKENS
            if max_context_tokens is not None
            else None
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
            base_url=config.openai_base_url, api_key=config.openai_api_key
        )
        self._provider_type = config.provider_type()
        self._metrics: list[RequestMetric] = []
        self._metrics_lock = threading.Lock()
        self._reported_reasoning_models: set[str] = set()

    def _apply_reasoning_options(self, kwargs: dict) -> dict:
        """按插件计划追加关闭或降档思考参数。"""
        request, plan = apply_thinking_options(
            kwargs,
            self._provider_type,
            self.config.disable_thinking,
        )
        response_format = request.get("response_format")
        response_format_type = (
            response_format.get("type") if isinstance(response_format, dict) else "none"
        )
        try:
            logger.info(
                "请求参数: provider=%s, model=%s, response_format=%s, reasoning=%s",
                self._provider_type,
                request.get("model") or "",
                response_format_type,
                plan.log_state,
            )
        except Exception:
            pass
        return request

    def metrics_checkpoint(self) -> int:
        """返回当前请求指标游标，供单文件统计隔离使用。"""
        with self._metrics_lock:
            return len(self._metrics)

    def metrics_summary(self, since: int = 0) -> dict[str, Any]:
        """汇总指标游标之后的所有请求。"""
        with self._metrics_lock:
            metrics = list(self._metrics[max(0, since) :])
        return summarize_request_metrics(metrics)

    def _record_metric(self, metric: RequestMetric) -> None:
        with self._metrics_lock:
            self._metrics.append(metric)

    def _report_detected_reasoning(
        self,
        model: str,
        reasoning_tokens: Optional[int],
        has_reasoning_content: bool = False,
    ) -> None:
        """响应确认使用了推理时提示；同一模型每次运行只提示一次。"""
        if not has_reasoning_content and (
            reasoning_tokens is None or reasoning_tokens <= 0
        ):
            return

        display_model = model or "unknown"
        model_key = display_model.casefold()
        with self._metrics_lock:
            if model_key in self._reported_reasoning_models:
                return
            self._reported_reasoning_models.add(model_key)

        evidence = (
            f"reasoning_tokens={reasoning_tokens}"
            if reasoning_tokens is not None and reasoning_tokens > 0
            else "reasoning_content=present"
        )
        notice = detected_reasoning_notice(
            display_model,
            self._provider_type,
            self.config.disable_thinking,
            evidence,
        )
        try:
            if notice.level == "warning":
                logger.warning(notice.message)
                rich_print(f"[bold yellow]⚠️ {escape(notice.message)}[/bold yellow]")
            else:
                logger.info(notice.message)
                rich_print(f"[bold cyan]🧠 {escape(notice.message)}[/bold cyan]")
        except Exception:
            pass

    def _observe_successful_response(
        self,
        request: dict[str, Any],
        response: Any,
        started_at: float,
        ended_at: float,
        input_chars: int,
        input_words: int,
    ) -> None:
        """记录成功响应；观测失败不得影响调用方取得模型响应。"""
        try:
            elapsed = ended_at - started_at
            usage = get_response_usage(response)
            finish_reason = _response_finish_reason(response)
            content_chars = _response_content_size(response)
            reasoning_tokens = (
                usage["reasoning_tokens"]
                if isinstance(usage["reasoning_tokens"], int)
                else None
            )
            self._report_detected_reasoning(
                str(request.get("model") or ""),
                reasoning_tokens,
                _response_has_reasoning_content(response),
            )
            self._record_metric(
                RequestMetric(
                    started_at=started_at,
                    ended_at=ended_at,
                    latency=elapsed,
                    success=True,
                    completion_tokens=(
                        usage["completion_tokens"]
                        if isinstance(usage["completion_tokens"], int)
                        else None
                    ),
                    finish_reason=finish_reason,
                    content_chars=content_chars,
                    error_type=None,
                    prompt_tokens=(
                        usage["prompt_tokens"]
                        if isinstance(usage["prompt_tokens"], int)
                        else None
                    ),
                    total_tokens=(
                        usage["total_tokens"]
                        if isinstance(usage["total_tokens"], int)
                        else None
                    ),
                )
            )
            logger.info(
                "请求完成: model=%s, request_id=%s, latency=%.2fs, "
                "input_chars=%s, input_words=%s, prompt_tokens=%s, "
                "completion_tokens=%s, total_tokens=%s, reasoning_tokens=%s, "
                "finish_reason=%s, content_chars=%s",
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
                reasoning_tokens if reasoning_tokens is not None else "unknown",
                finish_reason or "unknown",
                content_chars,
            )
        except Exception as exc:
            try:
                logger.warning(
                    "响应观测失败，已保留模型响应: model=%s, error_type=%s",
                    request.get("model", ""),
                    type(exc).__name__,
                )
            except Exception:
                pass

    def create_chat_completion(self, **kwargs):
        """统一创建聊天补全请求。"""
        request = self._apply_reasoning_options(kwargs)
        try:
            input_chars, input_words = _request_text_size(request.get("messages"))
        except Exception:
            input_chars, input_words = 0, 0
        start_time = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            ended_at = time.perf_counter()
            try:
                self._record_metric(
                    RequestMetric(
                        started_at=start_time,
                        ended_at=ended_at,
                        latency=ended_at - start_time,
                        success=False,
                        completion_tokens=None,
                        finish_reason=None,
                        content_chars=0,
                        error_type=type(exc).__name__,
                    )
                )
            except Exception:
                pass
            try:
                logger.warning(
                    "请求失败: model=%s, latency=%.2fs, input_chars=%s, "
                    "input_words=%s, error_type=%s",
                    request.get("model", ""),
                    ended_at - start_time,
                    input_chars,
                    input_words,
                    type(exc).__name__,
                )
            except Exception:
                pass
            raise

        self._observe_successful_response(
            request=request,
            response=response,
            started_at=start_time,
            ended_at=time.perf_counter(),
            input_chars=input_chars,
            input_words=input_words,
        )
        return response

    def close(self) -> None:
        """释放当前 adapter 拥有的底层连接。"""
        self._client.close()
