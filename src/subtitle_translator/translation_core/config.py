import os
from dataclasses import dataclass
from typing import Optional

# 语言代码映射表
LANGUAGE_MAPPING = {
    "zh": "简体中文",
    "zh-cn": "简体中文", 
    "zh-tw": "繁体中文",
    "ja": "日文",
    "japanese": "日文",
    "en": "English",
    "english": "English",
    "ko": "韩文",
    "korean": "韩文",
    "fr": "法文",
    "french": "法文",
    "de": "德文",
    "german": "德文",
    "es": "西班牙文",
    "spanish": "西班牙文",
    "pt": "葡萄牙文",
    "portuguese": "葡萄牙文",
    "ru": "俄文",
    "russian": "俄文",
    "it": "意大利文",
    "italian": "意大利文",
    "ar": "阿拉伯文",
    "arabic": "阿拉伯文",
    "th": "泰文",
    "thai": "泰文",
    "vi": "越南文",
    "vietnamese": "越南文",
}

def get_target_language(lang_code: str) -> str:
    """
    将语言代码转换为目标语言名称
    
    Args:
        lang_code: 语言代码，如 'zh', 'ja', 'en' 等
        
    Returns:
        str: 对应的语言名称，如 '简体中文', '日文', 'English' 等
        
    Raises:
        ValueError: 如果语言代码不支持
    """
    lang_code = lang_code.lower().strip()
    if lang_code in LANGUAGE_MAPPING:
        return LANGUAGE_MAPPING[lang_code]
    else:
        supported_langs = ", ".join(sorted(set(LANGUAGE_MAPPING.keys())))
        raise ValueError(f"不支持的语言代码: '{lang_code}'。支持的语言代码: {supported_langs}")

@dataclass
class SubtitleConfig:
    """字幕处理配置类"""
    # API配置 - 使用默认值，在__post_init__中重新读取
    openai_base_url: str = ""
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"  # 兼容性字段，作为默认值
    
    # 各功能模型配置
    split_model: str = "gpt-4o-mini"      # 断句模型
    summary_model: str = "gpt-4o-mini"    # 总结模型
    translation_model: str = "gpt-4o"     # 翻译模型  
    
    # 处理配置
    target_language: str = "简体中文"  # 默认值，可通过 set_target_language 方法修改
    max_word_count_english: int = 14
    thread_num: int = 18
    batch_size: int = 20
    
    # 功能开关
    need_reflect: bool = False
    
    def set_target_language(self, lang_code: str) -> None:
        """
        设置目标语言
        
        Args:
            lang_code: 语言代码，如 'zh', 'ja', 'en' 等
        """
        self.target_language = get_target_language(lang_code)
    
    def __post_init__(self):
        """验证配置并重新读取环境变量"""
        # 重新读取环境变量，确保.env文件已加载
        self.openai_base_url = os.getenv('OPENAI_BASE_URL', '')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.llm_model = os.getenv('LLM_MODEL', self.llm_model)
        
        # 读取各功能模型配置，如果未设置则使用 llm_model 作为默认值
        self.split_model = os.getenv('SPLIT_MODEL', self.llm_model)
        self.summary_model = os.getenv('SUMMARY_MODEL', self.llm_model)
        self.translation_model = os.getenv('TRANSLATION_MODEL', self.llm_model)
        
        # 从环境变量读取目标语言（如果设置了的话）
        env_target_lang = os.getenv('TARGET_LANGUAGE')
        if env_target_lang:
            try:
                self.set_target_language(env_target_lang)
            except ValueError:
                # 如果环境变量中的语言代码无效，保持默认值
                pass
        
        if not self.openai_base_url or not self.openai_api_key:
            error_msg = f"环境变量验证失败:\n"
            error_msg += f"  OPENAI_BASE_URL = '{self.openai_base_url}' (长度: {len(self.openai_base_url)})\n"
            error_msg += f"  OPENAI_API_KEY = '{self.openai_api_key[:20]}...' (长度: {len(self.openai_api_key)})\n"
            error_msg += f"  LLM_MODEL = '{self.llm_model}'\n"
            error_msg += f"  SPLIT_MODEL = '{self.split_model}'\n"
            error_msg += f"  SUMMARY_MODEL = '{self.summary_model}'"
            error_msg += f"  TRANSLATION_MODEL = '{self.translation_model}'\n"
            raise ValueError(error_msg)

# 文件相关常量
SRT_SUFFIX = ".srt"
OUTPUT_SUFFIX = "_zh.srt"
EN_OUTPUT_SUFFIX = "_en.srt"

# 延迟创建默认配置实例
def get_default_config() -> SubtitleConfig:
    """获取默认配置实例"""
    return SubtitleConfig() 