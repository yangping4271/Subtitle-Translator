"""统一异常定义模块

所有项目异常都继承自 SubtitleTranslatorError 基类，
支持携带错误信息和用户建议。
"""


class SubtitleTranslatorError(Exception):
    """字幕翻译工具所有异常的基类

    Args:
        message: 错误描述信息
        suggestion: 可选的用户操作建议
    """
    def __init__(self, message: str, suggestion: str = ""):
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)

    def __str__(self):
        return self.message


class OpenAIAPIError(SubtitleTranslatorError):
    """OpenAI API 调用相关错误"""
    pass


class SmartSplitError(SubtitleTranslatorError):
    """智能断句异常 - 用于 LLM 调用失败或返回格式错误"""
    pass


class TranslationError(SubtitleTranslatorError):
    """翻译异常 - 用于翻译过程中的错误"""
    pass


class SummaryError(SubtitleTranslatorError):
    """内容分析异常 - 用于总结过程中的错误"""
    pass


class EmptySubtitleError(SubtitleTranslatorError):
    """空字幕文件异常 - 用于 SRT 文件为空的情况"""
    pass


class SubtitleProcessError(SubtitleTranslatorError):
    """字幕处理通用异常 - 用于其他字幕处理相关错误"""
    pass
