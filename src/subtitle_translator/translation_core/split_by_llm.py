import math
import re
from typing import List, Optional

from .prompts import SPLIT_SYSTEM_PROMPT
from .config import SubtitleConfig
from .llm_client import ModelAdapter
from .segmentation_rules import count_words, split_by_end_marks
from .utils.errors import extract_error_message, get_error_suggestions
from .utils.api import validate_api_response
from ..logger import setup_logger

logger = setup_logger("split_by_llm")


def split_by_llm(
    text: str,
    config: SubtitleConfig,
    llm: ModelAdapter,
    model: Optional[str] = None,
    max_word_count_english: int = 14,
    max_retries: int = 3,
    batch_index: Optional[int] = None,
) -> List[str]:
    """使用LLM拆分句子。"""
    logger.info(f"📝 处理文本: 共{count_words(text)}个单词")

    if model is None:
        model = config.split_model

    system_prompt = SPLIT_SYSTEM_PROMPT.format(
        max_word_count_english=max_word_count_english
    )

    user_prompt = f"Please use multiple <br> tags to separate the following sentence. Make sure to preserve all spaces and punctuation exactly as they appear in the original text:\n{text}"

    try:
        response = llm.create_chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            timeout=80,
        )

        result = validate_api_response(response)
        if not result:
            raise Exception("API返回为空")
        if config.log_raw_payloads:
            logger.debug(f"API返回结果: \n\n{result}\n")
        else:
            logger.info("API返回结果: %s 字符（原文日志已关闭）", len(result))

        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)

        result = re.sub(r"\n+", "", result)

        sentences = result.split("<br>")

        sentences = [seg.strip() for seg in sentences if seg.strip()]

        tolerance_threshold = int(max_word_count_english * config.tolerance_multiplier)
        warning_threshold = int(max_word_count_english * config.warning_multiplier)
        max_threshold = int(max_word_count_english * config.max_multiplier)
        stats = dict.fromkeys(
            ("normal", "tolerated", "optimized", "forced", "rejected"), 0
        )
        new_sentences = []
        for sentence in sentences:
            for segment in split_by_end_marks(sentence, require_space=True):
                word_count = count_words(segment)
                parts = [segment]
                category = (
                    "normal" if word_count <= max_word_count_english else "tolerated"
                )
                if (
                    word_count > tolerance_threshold
                    and word_count > max_word_count_english
                ):
                    parts = aggressive_split(segment, max_word_count_english)
                    if len(parts) > 1:
                        category = "optimized"
                    elif word_count > warning_threshold:
                        parts = fallback_split(
                            segment, max_word_count_english, warning_threshold
                        )
                        category = (
                            "rejected" if word_count > max_threshold else "forced"
                        )
                new_sentences.extend(parts)
                stats[category] += 1
        sentences = new_sentences
        logger.info(
            "📊 断句质量: 正常=%s, 容忍=%s, 优化=%s, 强制=%s, 严重超标=%s",
            *stats.values(),
        )

        # 验证结果
        word_count = count_words(text)
        expected_segments = word_count / max_word_count_english
        actual_segments = len(sentences)

        if actual_segments < expected_segments * 0.9:
            logger.warning(
                f"⚠️ 断句数量不足：预期 {expected_segments:.1f}，实际 {actual_segments}"
            )

        batch_prefix = f"[批次{batch_index}]" if batch_index else ""
        logger.info(f"✅ {batch_prefix} 断句完成: {len(sentences)} 个句子")
        return sentences

    except Exception as e:
        if max_retries > 0:
            logger.warning(
                f"API调用失败，第{4 - max_retries}次重试: {extract_error_message(str(e))}"
            )
            return split_by_llm(
                text,
                config,
                llm,
                model,
                max_word_count_english,
                max_retries - 1,
                batch_index,
            )
        else:
            error_msg = extract_error_message(str(e))
            logger.error(f"智能断句失败: {error_msg}")

            suggestions = get_error_suggestions(str(e), model)

            from ..exceptions import SmartSplitError

            raise SmartSplitError(error_msg, suggestions)


def aggressive_split(text: str, max_words: int) -> List[str]:
    """智能分割：基于语义边界的拆分。"""
    words = text.split()
    word_count = len(words)

    if word_count <= max_words:
        return [text]

    logger.info(f"🔧 尝试智能分割: {word_count}字 -> 目标≤{max_words}字")

    # 优先级设计原则：保护语义完整性，避免破坏不可分割的语义单元
    split_candidates = []

    punctuation_rules = ((".!?", 10, "句号"), (";:", 9, "分隔"), (",", 8, "逗号"))
    word_rules = (
        ({"and", "but", "or", "so", "yet", "nor"}, 7, "并列连词"),
        (
            {
                "because",
                "although",
                "though",
                "unless",
                "since",
                "while",
                "whereas",
                "if",
                "when",
                "before",
                "after",
            },
            6,
            "从属连词",
        ),
        (
            {"that", "which", "who", "whom", "whose", "where", "when", "whether"},
            5,
            "关系词",
        ),
    )
    for i in range(3, word_count - 2):
        word = words[i]
        for marks, priority, reason in punctuation_rules:
            if word.endswith(tuple(marks)):
                split_candidates.append((i + 1, priority, f"{reason}'{word[-1]}'"))
        normalized = word.lower().strip(",.!?")
        for candidates, priority, reason in word_rules:
            if normalized in candidates:
                split_candidates.append((i, priority, f"{reason}'{normalized}'"))

    if not split_candidates:
        return [text]

    best_pos, priority, reason = min(
        split_candidates,
        key=lambda candidate: (-candidate[1], abs(candidate[0] - word_count // 2)),
    )
    logger.info("语义分割: %s (优先级%s)", reason, priority)
    result = []
    for part in (" ".join(words[:best_pos]), " ".join(words[best_pos:])):
        if count_words(part) > int(max_words * 1.5):
            result.extend(aggressive_split(part, max_words))
        else:
            result.append(part)
    return result


def fallback_split(
    text: str, max_words: int, warning_threshold: int = None
) -> List[str]:
    """降级分割（兜底方案）：在理想切分点附近寻找语义边界。"""
    if warning_threshold is None:
        warning_threshold = int(max_words * 1.5)
    words = text.split()
    word_count = len(words)
    num_segments = math.ceil(word_count / max_words)
    if num_segments == 1:
        return [text]

    def boundary_score(pos: int) -> int:
        if words[pos - 1].endswith((".", "!", "?")):
            return 10
        if words[pos - 1].endswith((",", ";", ":")):
            return 8
        if words[pos].lower() in {"and", "but", "or", "so", "because", "when", "while"}:
            return 6
        return 1

    # 在各等分点前后五词内，优先标点和连接词；同分时取最近位置。
    boundaries = [0]
    for index in range(1, num_segments):
        ideal = int(word_count / num_segments * index)
        boundaries.append(
            max(
                range(max(1, ideal - 5), min(word_count - 1, ideal + 5) + 1),
                key=lambda pos: (boundary_score(pos), -abs(pos - ideal)),
            )
        )
    boundaries.append(word_count)

    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        part = words[start:end]
        if not part:
            continue
        if count_words(" ".join(part)) > warning_threshold:
            middle = len(part) // 2
            result.extend((" ".join(part[:middle]), " ".join(part[middle:])))
        else:
            result.append(" ".join(part))
    logger.info("降级分割: %s词 -> %s段", word_count, len(result))
    return result
