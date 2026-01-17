# -*- coding: utf-8 -*-
"""
错误处理工具模块

提供统一的错误信息提取和建议生成功能。
"""


def extract_error_message(error_str: str) -> str:
    """
    提取错误信息中的核心内容

    Args:
        error_str: 原始错误字符串

    Returns:
        str: 简化后的错误信息
    """
    # 提取 API 错误信息
    if "Error code:" in error_str and "message" in error_str:
        try:
            import json
            import re

            # 查找 JSON 部分
            json_match = re.search(r'\{.*\}', error_str)
            if json_match:
                try:
                    error_data = json.loads(json_match.group())
                    if "error" in error_data and "message" in error_data["error"]:
                        return error_data["error"]["message"]
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

    # 根据关键词返回简化的错误信息
    error_patterns = {
        "is not a valid model ID": "模型不存在或不可用",
        "401": "API密钥无效或已过期",
        "Unauthorized": "API密钥无效或已过期",
        "403": "API访问被拒绝",
        "Forbidden": "API访问被拒绝",
        "429": "API调用频率限制",
        "rate limit": "API调用频率限制",
        "timeout": "请求超时",
        "connection": "网络连接失败",
    }

    error_str_lower = error_str.lower()
    for pattern, message in error_patterns.items():
        if pattern.lower() in error_str_lower:
            return message

    # 返回前50个字符作为简化错误信息
    return error_str[:50] + ("..." if len(error_str) > 50 else "")


def get_error_suggestions(error_str: str, model: str) -> str:
    """
    根据错误类型返回针对性建议

    Args:
        error_str: 原始错误字符串
        model: 使用的模型名称

    Returns:
        str: 针对性的建议信息
    """
    error_str_lower = error_str.lower()

    suggestion_patterns = [
        ("is not a valid model ID", f"检查模型名称 '{model}' 是否正确，或更换其他可用模型"),
        ("401", "检查 API 密钥是否正确设置"),
        ("Unauthorized", "检查 API 密钥是否正确设置"),
        ("403", "检查 API 密钥权限或账户状态"),
        ("429", "稍后重试，或检查 API 调用频率限制"),
        ("rate limit", "稍后重试，或检查 API 调用频率限制"),
        ("timeout", "检查网络连接，或尝试使用更快的模型"),
        ("connection", "检查网络连接和 API 端点设置"),
    ]

    for pattern, suggestion in suggestion_patterns:
        if pattern.lower() in error_str_lower:
            return f"建议：{suggestion}"

    return "建议：检查网络连接、API 密钥和模型配置"
