import os
from dataclasses import dataclass, field
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
    """将语言代码转换为目标语言名称"""
    if not lang_code or not isinstance(lang_code, str):
        raise ValueError(f"语言代码不能为空或非字符串类型: '{lang_code}'")

    lang_code = lang_code.lower().strip()
    if lang_code in LANGUAGE_MAPPING:
        return LANGUAGE_MAPPING[lang_code]

    return _build_language_error(lang_code)


def _build_language_error(lang_code: str) -> str:
    """构建语言代码错误信息"""
    language_groups = {
        "中文": ["zh", "zh-cn", "zh-tw"],
        "亚洲语言": ["ja", "ko", "th", "vi"],
        "欧洲语言": ["en", "fr", "de", "es", "pt", "it", "ru"],
        "其他语言": ["ar"]
    }

    error_msg = f"❌ 不支持的语言代码: '{lang_code}'\n\n🌍 支持的语言代码:\n"
    for group_name, codes in language_groups.items():
        group_codes = [code for code in codes if code in LANGUAGE_MAPPING]
        if group_codes:
            error_msg += f"\n📂 {group_name}:\n"
            for code in group_codes:
                lang_name = LANGUAGE_MAPPING[code]
                error_msg += f"   {code:6} -> {lang_name}\n"

    suggestions = {
        "jp": ["ja"], "kr": ["ko"], "cn": ["zh", "zh-cn"],
        "chinese": ["zh", "zh-cn"], "japanese": ["ja"], "korean": ["ko"],
        "english": ["en"], "french": ["fr"], "german": ["de"],
        "spanish": ["es"], "portuguese": ["pt"], "russian": ["ru"],
        "italian": ["it"], "arabic": ["ar"], "thai": ["th"],
        "vietnamese": ["vi"],
    }

    similar_codes = suggestions.get(lang_code, [])
    if not similar_codes:
        for supported_code in LANGUAGE_MAPPING.keys():
            if (lang_code in supported_code or supported_code in lang_code or
                abs(len(lang_code) - len(supported_code)) <= 1):
                similar_codes.append(supported_code)

    if similar_codes:
        error_msg += f"\n💡 您是否想要使用: {', '.join(similar_codes[:3])}"

    error_msg += f"\n\n📊 总计支持 {len(set(LANGUAGE_MAPPING.values()))} 种语言，{len(LANGUAGE_MAPPING)} 个语言代码"
    raise ValueError(error_msg)


def _read_positive_int_env(env_name: str, current_value: int) -> int:
    """读取正整数环境变量，非法值时保留当前值。"""
    raw_value = os.getenv(env_name)
    if not raw_value:
        return current_value

    try:
        parsed = int(raw_value)
    except ValueError:
        return current_value

    return max(1, parsed)

@dataclass
class SubtitleConfig:
    """字幕处理配置类"""
    openai_base_url: str = ""
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    split_model: str = "gpt-4o-mini"
    translation_model: str = "gpt-4o"

    target_language: str = "简体中文"
    max_word_count_english: int = 19
    thread_num: int = 18

    min_batch_sentences: int = 15
    max_batch_sentences: int = 25
    target_batch_sentences: int = 20

    tolerance_multiplier: float = 1.2
    warning_multiplier: float = 1.5
    max_multiplier: float = 2.0
    lm_studio_ttl: Optional[int] = None
    lm_studio_unload_on_complete: bool = False

    need_reflect: bool = False

    terminology: Optional[dict] = None

    _skip_env_load: bool = field(default=False, repr=False)

    def set_target_language(self, lang_code: str) -> None:
        """设置目标语言"""
        self.target_language = get_target_language(lang_code)

    def __post_init__(self):
        """验证配置并重新读取环境变量"""
        if self._skip_env_load:
            return

        self.openai_base_url = os.getenv('OPENAI_BASE_URL', '')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.llm_model = os.getenv('LLM_MODEL', self.llm_model)

        self.split_model = os.getenv('SPLIT_MODEL', self.llm_model)
        self.translation_model = os.getenv('TRANSLATION_MODEL', self.llm_model)

        self.thread_num = _read_positive_int_env('THREAD_NUM', self.thread_num)
        self.max_word_count_english = _read_positive_int_env(
            'MAX_WORD_COUNT_ENGLISH', self.max_word_count_english
        )
        self.min_batch_sentences = _read_positive_int_env(
            'MIN_BATCH_SENTENCES', self.min_batch_sentences
        )
        self.max_batch_sentences = _read_positive_int_env(
            'MAX_BATCH_SENTENCES', self.max_batch_sentences
        )
        self.target_batch_sentences = _read_positive_int_env(
            'TARGET_BATCH_SENTENCES', self.target_batch_sentences
        )

        if self.min_batch_sentences > self.max_batch_sentences:
            self.min_batch_sentences = self.max_batch_sentences

        if not (self.min_batch_sentences <= self.target_batch_sentences <= self.max_batch_sentences):
            self.target_batch_sentences = min(
                max(self.target_batch_sentences, self.min_batch_sentences),
                self.max_batch_sentences,
            )

        env_lm_studio_ttl = os.getenv('LM_STUDIO_TTL')
        if env_lm_studio_ttl:
            try:
                self.lm_studio_ttl = max(1, int(env_lm_studio_ttl))
            except ValueError:
                pass

        env_unload_on_complete = os.getenv('LM_STUDIO_UNLOAD_ON_COMPLETE')
        if env_unload_on_complete:
            self.lm_studio_unload_on_complete = env_unload_on_complete.strip().lower() in {
                "1", "true", "yes", "on"
            }

        env_target_lang = os.getenv('TARGET_LANGUAGE')
        if env_target_lang:
            try:
                self.set_target_language(env_target_lang)
            except ValueError:
                pass

        if not self.openai_base_url:
            missing = []
            if not self.openai_base_url:
                missing.append("OPENAI_BASE_URL")
            raise ValueError(
                f"缺少必需的环境变量: {', '.join(missing)}。"
                f"请运行 'translate init' 初始化配置。"
            )

# 延迟创建默认配置实例
def get_default_config() -> SubtitleConfig:
    """获取默认配置实例"""
    return SubtitleConfig() 
