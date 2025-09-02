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

        # 验证句子长度
        new_sentences = []
        long_sentence_count = 0
        super_long_count = 0
        
        for sentence in sentences:
            # 首先按结束标记拆分句子
            segments = split_by_end_marks(sentence)
            
            # 对每个分段进行长度检查
            for segment in segments:
                threshold = max_word_count_english + 5
                word_count = count_words(segment)
                
                if max_word_count_english < word_count < threshold:
                    long_sentence_count += 1
                    logger.debug(f"⚠️ 长句: {word_count}字 - {segment[:30]}...")
                    new_sentences.append(segment)
                elif word_count > threshold:
                    logger.info(f"🔄 处理超长句: {word_count}字 - {segment[:30]}...")
                    
                    # 记录分割前的原始内容用于检查
                    original_segment = segment
                    
                    # 尝试切分句子
                    split_results = split_by_common_words(segment)
                    
                    # 检查是否实际分割成功
                    if len(split_results) > 1:
                        super_long_count += 1
                        logger.info(f"✅ 超长句分割成功: {len(split_results)} 个片段")
                    else:
                        logger.warning(f"⚠️ 超长句分割失败，保持原样")
                    
                    new_sentences.extend(split_results)
                else:
                    new_sentences.append(segment)
        
        sentences = new_sentences

        # 记录统计信息
        if long_sentence_count > 0:
            logger.info(f"📊 发现 {long_sentence_count} 个长句 (15-19字)")
        if super_long_count > 0:
            logger.info(f"✂️ 成功分割 {super_long_count} 个超长句 (>19字)")

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
    for i, segment in enumerate(result, 1):
        logger.info(f"   片段{i}({count_words(segment)}字): {segment}")
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