# -*- coding: utf-8 -*-
"""
API 工具模块

提供 OpenAI API 响应验证等通用功能。
"""
from typing import Any


def validate_api_response(response: Any, context: str = "") -> str:
    """
    验证 API 响应并返回内容

    Args:
        response: API 响应对象
        context: 上下文信息，用于错误消息

    Returns:
        str: 响应内容

    Raises:
        ValueError: 如果响应格式异常或内容为空
    """
    prefix = f"{context} " if context else ""

    # 检查是否是字符串错误响应
    if isinstance(response, str):
        raise ValueError(f"{prefix}API调用失败: {response}")

    # 检查是否有 choices 属性
    if not hasattr(response, 'choices') or not response.choices:
        raise ValueError(f"{prefix}API响应格式异常：缺少choices属性")

    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{prefix}API返回空内容")
    return content
