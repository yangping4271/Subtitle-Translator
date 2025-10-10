import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from .data import SubtitleSegment
from .prompts import SPLIT_SYSTEM_PROMPT
from .config import SubtitleConfig, get_default_config
from ..logger import setup_logger

logger = setup_logger("split_by_llm")

def _extract_error_message(error_str: str) -> str:
    """提取错误信息中的核心内容"""
    # 提取 API 错误信息
    if "Error code:" in error_str and "message" in error_str:
        try:
            # 尝试提取 JSON 中的 message 字段
            import json
            import re
            
            # 查找 JSON 部分
            json_match = re.search(r'\{.*\}', error_str)
            if json_match:
                try:
                    error_data = json.loads(json_match.group())
                    if "error" in error_data and "message" in error_data["error"]:
                        return error_data["error"]["message"]
                except:
                    pass
        except:
            pass
    
    # 如果无法解析 JSON，返回简化的错误信息
    if "is not a valid model ID" in error_str:
        return "模型不存在或不可用"
    elif "401" in error_str or "Unauthorized" in error_str:
        return "API密钥无效或已过期"
    elif "403" in error_str or "Forbidden" in error_str:
        return "API访问被拒绝"
    elif "429" in error_str or "rate limit" in error_str.lower():
        return "API调用频率限制"
    elif "timeout" in error_str.lower():
        return "请求超时"
    elif "connection" in error_str.lower():
        return "网络连接失败"
    else:
        # 返回前50个字符作为简化错误信息
        return error_str[:50] + ("..." if len(error_str) > 50 else "")

def _get_error_suggestions(error_str: str, model: str) -> str:
    """根据错误类型返回针对性建议"""
    if "is not a valid model ID" in error_str:
        return f"💡 建议：检查模型名称 '{model}' 是否正确，或更换其他可用模型"
    elif "401" in error_str or "Unauthorized" in error_str:
        return "💡 建议：检查 API 密钥是否正确设置"
    elif "403" in error_str:
        return "💡 建议：检查 API 密钥权限或账户状态"
    elif "429" in error_str or "rate limit" in error_str.lower():
        return "💡 建议：稍后重试，或检查 API 调用频率限制"
    elif "timeout" in error_str.lower():
        return "💡 建议：检查网络连接，或尝试使用更快的模型"
    elif "connection" in error_str.lower():
        return "💡 建议：检查网络连接和 API 端点设置"
    else:
        return "💡 建议：检查网络连接、API 密钥和模型配置"

def count_words(text: str) -> int:
    """
    统计文本中英文单词数
    Args:
        text: 输入文本，英文
    Returns:
        int: 英文单词数
    """
    english_text = re.sub(r'[\u4e00-\u9fff]', ' ', text)
    english_words = english_text.strip().split()
    return len(english_words)

def split_by_end_marks(sentence: str) -> List[str]:
    """
    按明确的句子结束标记拆分句子（简化版）
    
    Args:
        sentence: 需要拆分的句子
        
    Returns:
        List[str]: 拆分后的句子列表
    """
    # 只处理明确的句子结束标记，避免过度分割
    end_marks = [". ", "! ", "? "]
    positions = []
    
    # 查找句子结束标记的位置
    for mark in end_marks:
        start = 0
        while True:
            pos = sentence.find(mark, start)
            if pos == -1:
                break
            # 确保不是小数点
            if mark == ". " and pos > 0 and sentence[pos-1].isdigit():
                start = pos + 1
                continue
            positions.append(pos + 1)  # 标点后的位置
            start = pos + 1
    
    # 如果没有找到结束标记，返回原句子
    if not positions:
        return [sentence]
    
    # 执行分割
    positions.sort()
    segments = []
    start = 0
    
    for pos in positions:
        segment = sentence[start:pos].strip()
        # 确保每段至少有3个单词才分割
        if segment and count_words(segment) >= 3:
            segments.append(segment)
            start = pos
    
    # 处理最后一段
    last_segment = sentence[start:].strip()
    if last_segment:
        if segments and count_words(last_segment) < 2:
            # 最后一段太短，合并到前一段
            segments[-1] += " " + last_segment
        else:
            segments.append(last_segment)
    
    # 记录分割结果
    if len(segments) > 1:
        logger.info(f"✂️ 标点分割: {len(segments)}段")
    
    return segments if len(segments) > 1 else [sentence]

def split_by_llm(text: str,
                model: str = None,
                max_word_count_english: int = 14,
                max_retries: int = 3,
                batch_index: int = None) -> List[str]:
    """
    使用LLM拆分句子
    
    Args:
        text: 要拆分的文本
        model: 使用的语言模型，如果为None则使用配置中的断句模型
        max_word_count_english: 英文最大单词数
        max_retries: 最大重试次数
        batch_index: 批次索引，用于日志显示
        
    Returns:
        List[str]: 拆分后的句子列表
    """
    logger.info(f"📝 处理文本: 共{count_words(text)}个单词")
    
    # 初始化客户端
    config = SubtitleConfig()
    # 如果没有指定模型，使用配置中的断句模型
    if model is None:
        model = config.split_model
    
    client = OpenAI(
        base_url=config.openai_base_url,
        api_key=config.openai_api_key
    )
    
    # 使用系统提示词
    system_prompt = SPLIT_SYSTEM_PROMPT.replace("[max_word_count_english]", str(max_word_count_english))
    
    # 在用户提示中添加对空格的强调
    user_prompt = f"Please use multiple <br> tags to separate the following sentence. Make sure to preserve all spaces and punctuation exactly as they appear in the original text:\n{text}"

    try:
        # 调用API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            timeout=80
        )
        
        # 处理响应 - 添加类型检查
        if isinstance(response, str):
            logger.error(f"❌ API调用返回错误: {response}")
            raise Exception(f"API调用失败: {response}")
        
        # 检查response是否有choices属性
        if not hasattr(response, 'choices') or not response.choices:
            logger.error("❌ API响应格式异常：缺少choices属性")
            raise Exception("API响应格式异常")
        
        result = response.choices[0].message.content
        if not result:
            raise Exception("API返回为空")
        logger.debug(f"API返回结果: \n\n{result}\n")

        # 清理和分割文本 - 简化处理，保留原始格式
        result = re.sub(r'\n+', '', result)
        
        # 直接按<br>分割，保留原始格式和空格
        sentences = result.split("<br>")
        
        # 清理空白行，但保留内部空格
        sentences = [seg.strip() for seg in sentences if seg.strip()]

        # 四层防护机制验证句子长度
        # 动态计算阈值（基于配置的倍数参数）
        config = get_default_config()
        tolerance_threshold = int(max_word_count_english * config.tolerance_multiplier)      # 轻度容忍阈值
        warning_threshold = int(max_word_count_english * config.warning_multiplier)         # 警告阈值
        max_threshold = int(max_word_count_english * config.max_multiplier)                 # 最大阈值

        new_sentences = []
        stats = {
            'normal': 0,        # ≤ target
            'tolerated': 0,     # target < x ≤ tolerance
            'optimized': 0,     # tolerance < x ≤ warning（经过优化）
            'forced': 0,        # warning < x ≤ max（强制拆分）
            'rejected': 0       # > max（严重超标，强制多次拆分）
        }

        for sentence in sentences:
            # 首先按结束标记拆分句子
            segments = split_by_end_marks(sentence)

            # 对每个分段进行四层验证
            for segment in segments:
                word_count = count_words(segment)

                # 层级1：正常范围 (≤ target)
                if word_count <= max_word_count_english:
                    new_sentences.append(segment)
                    stats['normal'] += 1

                # 层级2：轻度容忍层 (target < x ≤ tolerance)
                elif word_count <= tolerance_threshold:
                    new_sentences.append(segment)
                    stats['tolerated'] += 1
                    logger.debug(f"✓ 轻度超标({word_count}/{max_word_count_english}字): {segment[:40]}...")

                # 层级3：强制优化层 (tolerance < x ≤ warning)
                elif word_count <= warning_threshold:
                    logger.info(f"🔧 尝试优化({word_count}/{max_word_count_english}字): {segment[:40]}...")
                    split_results = aggressive_split(segment, max_word_count_english)

                    if len(split_results) > 1:
                        stats['optimized'] += 1
                        logger.info(f"✅ 优化成功: 分为{len(split_results)}段")
                        new_sentences.extend(split_results)
                    else:
                        # 优化失败，但仍在可接受范围内
                        stats['tolerated'] += 1
                        logger.warning(f"⚠️ 优化失败，接受原句({word_count}字)")
                        new_sentences.append(segment)

                # 层级4：强制拆分层 (warning < x ≤ max)
                elif word_count <= max_threshold:
                    logger.warning(f"🔨 强制拆分({word_count}/{max_word_count_english}字): {segment[:40]}...")
                    split_results = force_equal_split(segment, max_word_count_english)
                    stats['forced'] += 1
                    new_sentences.extend(split_results)

                # 层级5：拒绝层 (> max)
                else:
                    logger.error(f"❌ 严重超标({word_count}/{max_word_count_english}字)，强制多次拆分: {segment[:40]}...")
                    split_results = force_equal_split(segment, max_word_count_english)
                    stats['rejected'] += 1
                    new_sentences.extend(split_results)

        sentences = new_sentences

        # 记录统计信息（使用动态阈值显示）
        logger.info(f"📊 断句质量统计:")
        logger.info(f"   ✅ 正常: {stats['normal']}句 (≤{max_word_count_english}字)")
        if stats['tolerated'] > 0:
            logger.info(f"   ✓ 轻度超标: {stats['tolerated']}句 ({max_word_count_english}-{tolerance_threshold}字)")
        if stats['optimized'] > 0:
            logger.info(f"   🔧 优化拆分: {stats['optimized']}句 ({tolerance_threshold}-{warning_threshold}字)")
        if stats['forced'] > 0:
            logger.warning(f"   🔨 强制拆分: {stats['forced']}句 ({warning_threshold}-{max_threshold}字)")
        if stats['rejected'] > 0:
            logger.error(f"   ❌ 严重超标: {stats['rejected']}句 (>{max_threshold}字)")

        # 验证结果
        word_count = count_words(text)
        expected_segments = word_count / max_word_count_english
        actual_segments = len(sentences)

        if actual_segments < expected_segments * 0.9:
            logger.warning(f"⚠️ 断句数量不足：预期 {expected_segments:.1f}，实际 {actual_segments}")

        batch_prefix = f"[批次{batch_index}]" if batch_index else ""
        logger.info(f"✅ {batch_prefix} 断句完成: {len(sentences)} 个句子")
        return sentences
        
    except Exception as e:
        if max_retries > 0:
            logger.warning(f"API调用失败，第{4-max_retries}次重试: {_extract_error_message(str(e))}")
            return split_by_llm(text, model, max_word_count_english, max_retries-1, batch_index)
        else:
            error_msg = _extract_error_message(str(e))
            logger.error(f"❌ 智能断句失败: {error_msg}")
            
            # 根据错误类型给出针对性建议
            suggestions = _get_error_suggestions(str(e), model)
            
            # 创建一个携带建议的自定义异常类型
            from .spliter import SmartSplitError
            raise SmartSplitError(error_msg, suggestions)
        
def split_by_common_words(text: str) -> List[str]:
    """
    改进的智能分割策略：依次尝试多种分割方法
    
    Args:
        text: 需要分割的句子
    Returns:
        分割后的句子列表
    """
    text = text.strip()
    words = text.split()
    
    # 如果句子太短，直接返回
    if len(words) < 8:
        return [text]
    
    logger.info(f"🔧 需要分割超长句: {count_words(text)}字 - {text[:50]}...")
    
    # 策略1: 使用 spaCy 语法分析分割
    try:
        from .spacy_splitter import spacy_split
        result = spacy_split(text)
        if result:
            logger.info(f"✅ 使用spaCy语法分割: {len(result)}段")
            for i, segment in enumerate(result, 1):
                logger.info(f"   片段{i}({count_words(segment)}字): {segment}")
            return result
        else:
            logger.debug("spaCy 未找到合适分割点")
    except ImportError:
        logger.debug("spaCy 模块导入失败")
    except Exception as e:
        logger.debug(f"spaCy 分割异常: {e}")
    
    # 策略2: 句末标点分割
    result = split_by_punctuation_optimized(text)
    if len(result) > 1:
        logger.info(f"✅ 使用标点分割: {len(result)}段")
        for i, segment in enumerate(result, 1):
            logger.info(f"   片段{i}({count_words(segment)}字): {segment}")
        return result
    else:
        logger.debug("标点分割未找到合适分割点")
    
    # 策略3: 改进的强制二分
    result = force_smart_split(text)
    logger.info(f"✅ 使用强制智能分割: {len(result)}段")
    return result

def split_by_punctuation_optimized(text: str) -> List[str]:
    """
    基于句末标点的优化分割
    只在明确的句子结束处分割，确保每段有足够长度
    """
    # 只处理明确的句子结束标记
    end_marks = [". ", "! ", "? "]
    positions = []
    
    # 查找句子结束标记
    for mark in end_marks:
        start = 0
        while True:
            pos = text.find(mark, start)
            if pos == -1:
                break
            # 检查不是小数点
            if mark == ". " and pos > 0 and text[pos-1].isdigit():
                start = pos + 1
                continue
            positions.append(pos + 1)  # 标点后的位置
            start = pos + 1
    
    if not positions:
        return [text]
    
    # 执行分割
    positions.sort()
    segments = []
    start = 0
    
    for pos in positions:
        segment = text[start:pos].strip()
        # 确保每段至少有5个单词
        if segment and count_words(segment) >= 5:
            segments.append(segment)
            start = pos
    
    # 处理最后一段
    last_segment = text[start:].strip()
    if last_segment:
        if segments and count_words(last_segment) < 3:
            # 最后一段太短，合并到前一段
            segments[-1] += " " + last_segment
        else:
            segments.append(last_segment)
    
    return segments if len(segments) > 1 else [text]

def force_smart_split(text: str) -> List[str]:
    """
    改进的强制分割策略
    依次尝试：标点位置 -> 连接词位置 -> 中间位置
    """
    words = text.split()
    mid_point = len(words) // 2
    search_range = 8  # 扩大搜索范围

    start_search = max(3, mid_point - search_range)
    end_search = min(len(words) - 3, mid_point + search_range)

    best_split = mid_point
    split_reason = "中间位置"

    # 优先级1: 寻找标点符号位置
    for i in range(start_search, end_search + 1):
        if i < len(words):
            word = words[i-1].rstrip()
            if word.endswith(('.', ',', ';', ':', '!', '?')):
                best_split = i
                split_reason = f"标点'{word[-1]}'"
                break

    # 优先级2: 寻找连接词位置
    if split_reason == "中间位置":
        connection_words = ["and", "but", "or", "so", "because", "when", "if", "while"]
        for i in range(start_search, end_search + 1):
            if i < len(words):
                word = words[i].lower().strip(",.!?")
                if word in connection_words:
                    best_split = i
                    split_reason = f"连接词'{word}'"
                    break

    # 执行分割
    first_part = " ".join(words[:best_split])
    second_part = " ".join(words[best_split:])

    logger.info(f"强制分割在{split_reason}处:")
    for i, segment in enumerate([first_part, second_part], 1):
        logger.info(f"   片段{i}({count_words(segment)}字): {segment}")
    return [first_part, second_part]


def aggressive_split(text: str, max_words: int) -> List[str]:
    """
    增强版智能分割：全句搜索最优分割点

    优先级排序：
    1. 句号/问号/感叹号（完整句子边界）
    2. 分号/冒号（强逻辑分隔）
    3. 逗号 + 并列连词（and/but/or/so）
    4. 从属连词（because/although/since/while等）
    5. 关系代词（that/which/who/where/when）
    6. 介词短语边界（of/in/on/at/with/for/by等）
    7. 强制等分（最后手段）

    Args:
        text: 需要分割的文本
        max_words: 最大单词数限制

    Returns:
        分割后的句子列表，每段尽量 ≤ max_words
    """
    words = text.split()
    word_count = len(words)

    # 如果已经满足要求，直接返回
    if word_count <= max_words:
        return [text]

    logger.info(f"🔧 尝试智能分割: {word_count}字 -> 目标≤{max_words}字")

    # 定义分割点候选和优先级
    split_candidates = []

    # 优先级1: 句子结束标记
    for i, word in enumerate(words):
        if i > 2 and i < word_count - 2:  # 避免太短的片段
            if word.rstrip().endswith(('.', '!', '?')):
                split_candidates.append((i + 1, 10, f"句号'{word[-1]}'"))

    # 优先级2: 分号/冒号
    for i, word in enumerate(words):
        if i > 2 and i < word_count - 2:
            if word.rstrip().endswith((';', ':')):
                split_candidates.append((i + 1, 9, f"分隔'{word[-1]}'"))

    # 优先级3: 逗号 + 并列连词
    coordinating_conj = ["and", "but", "or", "so", "yet", "nor"]
    for i in range(1, word_count - 2):
        if words[i - 1].rstrip().endswith(','):
            next_word = words[i].lower().strip(",.!?")
            if next_word in coordinating_conj:
                split_candidates.append((i, 8, f"逗号+'{next_word}'"))

    # 优先级4: 从属连词（在句中的位置）
    subordinating_conj = ["because", "although", "though", "unless", "since",
                          "while", "whereas", "if", "when", "before", "after"]
    for i in range(3, word_count - 2):
        word = words[i].lower().strip(",.!?")
        if word in subordinating_conj:
            split_candidates.append((i, 7, f"从属连词'{word}'"))

    # 优先级5: 关系代词（从句开始）
    relative_pronouns = ["that", "which", "who", "whom", "whose", "where", "when"]
    for i in range(3, word_count - 2):
        word = words[i].lower().strip(",.!?")
        if word in relative_pronouns:
            split_candidates.append((i, 6, f"关系词'{word}'"))

    # 优先级6: 介词短语（较长介词）
    prepositions = ["of", "in", "on", "at", "with", "for", "by", "from",
                   "about", "during", "through", "between", "among"]
    for i in range(max(3, word_count // 3), min(word_count - 2, word_count * 2 // 3)):
        word = words[i].lower().strip(",.!?")
        if word in prepositions:
            split_candidates.append((i, 5, f"介词'{word}'"))

    # 如果找到候选点，选择最优的
    if split_candidates:
        # 按优先级排序，同优先级选择最接近中点的
        mid_point = word_count // 2
        split_candidates.sort(key=lambda x: (-x[1], abs(x[0] - mid_point)))

        best_pos, priority, reason = split_candidates[0]

        # 执行分割
        first_part = " ".join(words[:best_pos]).strip()
        second_part = " ".join(words[best_pos:]).strip()

        logger.info(f"✅ 智能分割在{reason}处 (优先级{priority}):")
        logger.info(f"   片段1({count_words(first_part)}字): {first_part[:50]}...")
        logger.info(f"   片段2({count_words(second_part)}字): {second_part[:50]}...")

        # 递归处理仍然超长的片段
        result = []
        for part in [first_part, second_part]:
            if count_words(part) > max_words:
                result.extend(aggressive_split(part, max_words))
            else:
                result.append(part)
        return result

    # 没有找到合适的分割点，使用强制等分
    logger.warning("⚠️ 未找到语义边界，使用强制等分")
    return force_equal_split(text, max_words)


def force_equal_split(text: str, max_words: int) -> List[str]:
    """
    保底强制等分：确保每段 ≤ max_words

    在理想等分点附近寻找最近的语义边界（标点或连接词）
    如果找不到，则在词边界强制切分

    Args:
        text: 需要分割的文本
        max_words: 最大单词数限制

    Returns:
        分割后的句子列表，保证每段 ≤ max_words
    """
    words = text.split()
    word_count = len(words)

    # 计算需要分成几段
    import math
    num_segments = math.ceil(word_count / max_words)

    if num_segments == 1:
        return [text]

    logger.info(f"🔨 强制等分: {word_count}字 -> {num_segments}段 (每段≤{max_words}字)")

    # 计算理想分割点
    segment_size = word_count / num_segments
    ideal_points = [int(segment_size * i) for i in range(1, num_segments)]

    # 在每个理想点附近寻找最佳分割位置
    actual_splits = []
    search_range = 5  # 在理想点前后5个单词范围内搜索

    for ideal_pos in ideal_points:
        best_pos = ideal_pos
        best_score = 0

        start = max(1, ideal_pos - search_range)
        end = min(word_count - 1, ideal_pos + search_range)

        for i in range(start, end + 1):
            score = 0
            word = words[i - 1].rstrip()

            # 评分：标点优于连接词优于普通位置
            if word.endswith(('.', '!', '?')):
                score = 10
            elif word.endswith((',', ';', ':')):
                score = 8
            elif i < word_count and words[i].lower() in ["and", "but", "or", "so", "because", "when", "while"]:
                score = 6
            else:
                score = 1

            # 同等分数下，优先选择更接近理想点的
            if score > best_score or (score == best_score and abs(i - ideal_pos) < abs(best_pos - ideal_pos)):
                best_score = score
                best_pos = i

        actual_splits.append(best_pos)

    # 执行分割
    result = []
    start_idx = 0

    for split_pos in actual_splits:
        segment = " ".join(words[start_idx:split_pos]).strip()
        if segment:
            result.append(segment)
        start_idx = split_pos

    # 添加最后一段
    last_segment = " ".join(words[start_idx:]).strip()
    if last_segment:
        result.append(last_segment)

    # 输出分割结果
    logger.info(f"✅ 强制等分完成: {len(result)}段")
    for i, segment in enumerate(result, 1):
        seg_words = count_words(segment)
        logger.info(f"   片段{i}({seg_words}字): {segment[:50]}...")
        if seg_words > max_words:
            logger.warning(f"   ⚠️ 片段{i}仍超标，需再次分割")

    # 验证：如果仍有超标片段，递归处理
    final_result = []
    for segment in result:
        if count_words(segment) > max_words:
            # 简单二分
            seg_words = segment.split()
            mid = len(seg_words) // 2
            final_result.append(" ".join(seg_words[:mid]))
            final_result.append(" ".join(seg_words[mid:]))
        else:
            final_result.append(segment)

    return final_result