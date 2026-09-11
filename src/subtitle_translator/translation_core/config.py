import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse

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
        "其他语言": ["ar"],
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
        "jp": ["ja"],
        "kr": ["ko"],
        "cn": ["zh", "zh-cn"],
        "chinese": ["zh", "zh-cn"],
        "japanese": ["ja"],
        "korean": ["ko"],
        "english": ["en"],
        "french": ["fr"],
        "german": ["de"],
        "spanish": ["es"],
        "portuguese": ["pt"],
        "russian": ["ru"],
        "italian": ["it"],
        "arabic": ["ar"],
        "thai": ["th"],
        "vietnamese": ["vi"],
    }

    similar_codes = suggestions.get(lang_code, [])
    if not similar_codes:
        for supported_code in LANGUAGE_MAPPING.keys():
            if (
                lang_code in supported_code
                or supported_code in lang_code
                or abs(len(lang_code) - len(supported_code)) <= 1
            ):
                similar_codes.append(supported_code)

    if similar_codes:
        error_msg += f"\n💡 您是否想要使用: {', '.join(similar_codes[:3])}"

    error_msg += f"\n\n📊 总计支持 {len(set(LANGUAGE_MAPPING.values()))} 种语言，{len(LANGUAGE_MAPPING)} 个语言代码"
    raise ValueError(error_msg)


def validate_base_url(base_url: str) -> None:
    """验证 OpenAI-compatible API 端点已配置。"""
    if not base_url:
        raise ValueError(
            "缺少必需的环境变量: OPENAI_BASE_URL。请运行 'translate init' 初始化配置。"
        )


def resolve_configured_model(environ: Mapping[str, str]) -> str:
    """从环境变量解析模型名，不提供内置默认值。"""
    return _env_text(environ, "LLM_MODEL")


def validate_model_configuration(llm_model: str) -> None:
    """验证模型已手动配置。"""
    if not llm_model:
        raise ValueError(
            "缺少必需的模型配置: LLM_MODEL。请运行 'translate init' 初始化配置。"
        )


@dataclass
class SubtitleConfig:
    """字幕处理配置类"""

    openai_base_url: str = ""
    openai_api_key: str = ""
    llm_model: str = ""

    max_word_count_english: int = 19
    thread_num: int = 18

    min_batch_sentences: int = 15
    max_batch_sentences: int = 25
    target_batch_sentences: int = 20
    max_batch_words: int = 500

    tolerance_multiplier: float = 1.2
    warning_multiplier: float = 1.5
    max_multiplier: float = 2.0
    disable_thinking: bool = True
    log_raw_payloads: bool = False

    external_glossary_enabled: bool = True
    external_glossary_domains: tuple[str, ...] = ("programming", "tech", "education")
    external_glossary_max_terms: int = 12

    def provider_type(self) -> str:
        """根据 OpenAI-compatible Base URL 推断供应商类型。"""
        parsed = urlparse(self.openai_base_url)
        hostname = (parsed.hostname or "").lower()
        path_parts = {
            part.strip().lower() for part in parsed.path.split("/") if part.strip()
        }
        if hostname.endswith("deepseek.com") or "deepseek" in path_parts:
            return "deepseek"
        if hostname.endswith("openrouter.ai") or "openrouter" in path_parts:
            return "openrouter"
        if hostname.endswith("aliyuncs.com") and (
            "dashscope" in hostname or "compatible-mode" in path_parts
        ):
            return "dashscope"
        if hostname.endswith("bigmodel.cn"):
            return "zhipu"
        if hostname.endswith(("moonshot.cn", "moonshot.ai", "kimi.com", "kimi.ai")):
            return "kimi"
        if hostname.endswith(("minimax.io", "minimaxi.com")):
            return "minimax"
        if hostname.endswith("groq.com"):
            return "groq"
        if hostname.endswith("googleapis.com") and (
            "generativelanguage" in hostname or "generativelanguage" in path_parts
        ):
            return "google"
        if hostname.endswith("anthropic.com"):
            return "anthropic"
        if hostname.endswith("x.ai"):
            return "xai"
        if hostname.endswith(("volces.com", "bytepluses.com")):
            return "volcengine"
        if hostname == "api.openai.com":
            return "openai"
        return "custom"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "SubtitleConfig":
        """从环境变量创建并验证运行配置。"""
        env = os.environ if environ is None else environ
        defaults = cls()
        openai_base_url = env.get("OPENAI_BASE_URL", "")
        openai_api_key = env.get("OPENAI_API_KEY", "")
        validate_base_url(openai_base_url)

        llm_model = resolve_configured_model(env)
        validate_model_configuration(llm_model)
        external_glossary_domains = defaults.external_glossary_domains
        if env.get("EXTERNAL_GLOSSARY_DOMAINS"):
            parsed_domains = tuple(
                domain.strip()
                for domain in env["EXTERNAL_GLOSSARY_DOMAINS"].split(",")
                if domain.strip()
            )
            if parsed_domains:
                external_glossary_domains = parsed_domains

        return cls(
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            llm_model=llm_model,
            thread_num=_env_int(env, "THREAD_NUM", defaults.thread_num, minimum=1),
            max_batch_words=_env_int(env, "MAX_BATCH_WORDS", defaults.max_batch_words, minimum=1),
            disable_thinking=_env_bool(
                env,
                "DISABLE_THINKING",
                defaults.disable_thinking,
            ),
            log_raw_payloads=_env_bool(
                env,
                "LOG_RAW_PAYLOADS",
                defaults.log_raw_payloads,
            ),
            external_glossary_enabled=_env_bool(
                env,
                "EXTERNAL_GLOSSARY_ENABLED",
                defaults.external_glossary_enabled,
            ),
            external_glossary_domains=external_glossary_domains,
            external_glossary_max_terms=_env_int(
                env, "EXTERNAL_GLOSSARY_MAX_TERMS", defaults.external_glossary_max_terms
            ),
        )


def _env_text(environ: Mapping[str, str], name: str) -> str:
    return (environ.get(name) or "").strip()


def _env_bool(
    environ: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(environ: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(environ[name]))
    except (KeyError, ValueError):
        return default
