import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class SubtitleConfig:
    """字幕处理配置类"""
    # API配置 - 使用默认值，在__post_init__中重新读取
    openai_base_url: str = ""
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    
    # 处理配置
    target_language: str = "简体中文"
    max_word_count_english: int = 14
    thread_num: int = 18
    batch_size: int = 20
    
    # 功能开关
    need_reflect: bool = False
    
    def __post_init__(self):
        """验证配置并重新读取环境变量"""
        # 重新读取环境变量，确保.env文件已加载
        self.openai_base_url = os.getenv('OPENAI_BASE_URL', '')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.llm_model = os.getenv('LLM_MODEL', self.llm_model)
        
        if not self.openai_base_url or not self.openai_api_key:
            error_msg = f"环境变量验证失败:\n"
            error_msg += f"  OPENAI_BASE_URL = '{self.openai_base_url}' (长度: {len(self.openai_base_url)})\n"
            error_msg += f"  OPENAI_API_KEY = '{self.openai_api_key[:20]}...' (长度: {len(self.openai_api_key)})\n"
            error_msg += f"  LLM_MODEL = '{self.llm_model}'"
            raise ValueError(error_msg)

# 文件相关常量
SRT_SUFFIX = ".srt"
OUTPUT_SUFFIX = "_zh.srt"
EN_OUTPUT_SUFFIX = "_en.srt"

# 延迟创建默认配置实例
def get_default_config() -> SubtitleConfig:
    """获取默认配置实例"""
    return SubtitleConfig() 