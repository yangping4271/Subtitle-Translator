"""LLM 客户端统一封装

提供 OpenAI API 调用的单一入口，便于后续扩展和维护。
"""
from typing import Optional
from openai import OpenAI

from .config import SubtitleConfig


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
