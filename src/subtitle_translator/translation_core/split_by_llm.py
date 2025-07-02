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
    按句子结束标记（句号、感叹号等）拆分句子，只要有结束标记就分割
    
    Args:
        sentence: 需要拆分的句子
        
    Returns:
        List[str]: 拆分后的句子列表
    """
    # 定义结束标记，注意每个标记后面都加了空格
    end_marks = [". ", "! ", "? ", "... ", "…… ", "; ", "? ", "! "]
    positions = []
    
    # 查找所有结束标记的位置
    for mark in end_marks:
        start = 0
        while True:
            pos = sentence.find(mark, start)
            if pos == -1:
                break
            # 确保不是小数点
            if mark == ". " and pos > 0:
                prev_char = sentence[pos-1]
                if prev_char.isdigit():
                    start = pos + 1
                    continue
            positions.append((pos + len(mark.strip()), mark))
            start = pos + 1
    
    # 如果没有找到结束标记，返回原句子
    if not positions:
        return [sentence]
    
    # 按位置排序并执行分割
    positions.sort()
    segments = []
    start = 0
    
    for pos, mark in positions:
        segment = sentence[start:pos].strip()
        if segment:  # 只要不是空字符串就添加
            segments.append(segment)
        start = pos
    
    # 添加最后一段
    last_segment = sentence[start:].strip()
    if last_segment:
        segments.append(last_segment)
    
    # 记录日志
    if len(segments) > 1:
        logger.info(f"拆分优化: {' -- '.join(segments)}")
    
    return segments if segments else [sentence]

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
        
        # 处理响应
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
    在常见连接词处对句子进行分割

    Args:
        text: 需要分割的句子
    Returns:
        分割后的句子列表，如果智能分割失败则使用强制分割
    """
    # 定义在词语前面分割的常见词
    prefix_split_words = {
        # 连接词和介词
        "and", "or", "but", "if", "then", "because", "as", "until",
        "while", "what", "when", "where", "nor", "yet", "so", "for",
        "however", "moreover", "furthermore", "additionally", "besides",
        "therefore", "thus", "hence", "consequently",
        # 从句引导词
        "that", "which", "who", "whom", "whose", "why", "how",
        # 时间和条件
        "before", "after", "since", "while", "unless", "although",
        "though", "even though", "whereas", "whether",
        # 目的和结果
        "in order to", "so that", "such that", "in case", "provided that",
        # 对比和让步
        "despite", "in spite of", "rather than", "instead of",
        # 补充和递进
        "also", "besides", "in addition", "moreover", "furthermore",
        "similarly", "likewise", "meanwhile", "subsequently"
    }

    # 定义在词语后面分割的常见词
    suffix_split_words = {
        # 标点符号
        ".", ",", "!", "?", ":", ";", "...", "…",
        # 引号和括号
        "\"", "'", "'", "'", "'", "'", ")", "]", "}"
    }

    # 预处理文本
    text = text.strip()
    words = text.split()
    
    # 如果句子太短，直接返回
    if len(words) < 8:
        return [text]
        
    # 寻找分割点
    split_positions = []
    for i, word in enumerate(words):
        word_lower = word.lower().strip(",.!?")
        # 检查前缀分割词
        if i > 2 and i < len(words) - 2:  # 确保不在句子开头和结尾太近的位置
            # 检查单词和短语
            for prefix in prefix_split_words:
                if " " in prefix:  # 处理多词短语
                    if i + len(prefix.split()) <= len(words):
                        phrase = " ".join(words[i:i+len(prefix.split())]).lower()
                        if phrase == prefix:
                            split_positions.append(i)
                            break
                elif word_lower == prefix:  # 处理单个词
                    split_positions.append(i)
                    break
        
        # 检查后缀分割词
        if i > 2 and i < len(words) - 1:  # 确保不在句子开头太近的位置
            if word_lower in suffix_split_words:
                split_positions.append(i + 1)  # 在后缀词之后分割
    
    # 尝试智能分割
    result = []
    if split_positions:
        # 排序并去重分割点
        split_positions = sorted(list(set(split_positions)))
        
        # 执行分割
        start = 0
        for pos in split_positions:
            if pos - start >= 3:  # 确保每个分段至少有3个词
                segment = " ".join(words[start:pos])
                if segment:
                    result.append(segment)
                start = pos
        
        # 添加最后一个分段
        if start < len(words):
            last_segment = " ".join(words[start:])
            if last_segment:
                result.append(last_segment)
        
        # 检查智能分割结果
        if len(result) > 1:
            # 如果有多于两个分段，尝试合并最短的相邻分段
            while len(result) > 2:
                # 找出最短的相邻分段对
                min_length = float('inf')
                merge_index = 0
                
                for i in range(len(result) - 1):
                    current_len = count_words(result[i]) + count_words(result[i + 1])
                    if current_len < min_length:
                        min_length = current_len
                        merge_index = i
                
                # 合并找到的最短相邻分段
                merged_segment = result[merge_index] + " " + result[merge_index + 1]
                result = result[:merge_index] + [merged_segment] + result[merge_index + 2:]

            # 最终检查智能分割结果是否合理
            if (len(result) > 1 and 
                all(count_words(segment) >= 3 for segment in result)):
                
                # 检查分段是否平衡（差距不要太大）
                lengths = [count_words(segment) for segment in result]
                if max(lengths) <= min(lengths) * 3:  # 长度比例合理
                    logger.debug(f"智能分割: {' -- '.join(result)}")
                    return result

    # 智能分割失败，启用备用强制分割策略
    logger.warning(f"⚠️ 智能分割失败，启用备用强制分割策略: {text[:50]}...")
    
    # 备用策略1: 尝试在中间位置寻找较好的分割点
    mid_point = len(words) // 2
    best_split = mid_point
    
    # 在中间位置前后5个词的范围内寻找较好的分割点
    search_range = 5
    start_search = max(3, mid_point - search_range)
    end_search = min(len(words) - 3, mid_point + search_range)
    
    # 优先选择标点符号后的位置
    for i in range(start_search, end_search + 1):
        if i < len(words) and words[i-1].rstrip().endswith(('.', ',', ';', ':', '!', '?')):
            best_split = i
            break
    
    # 如果没找到标点符号，选择连接词前的位置
    if best_split == mid_point:
        for i in range(start_search, end_search + 1):
            if i < len(words):
                word = words[i].lower().strip(",.!?")
                if word in {"and", "or", "but", "so", "then", "however", "therefore"}:
                    best_split = i
                    break
    
    # 执行备用分割
    first_part = " ".join(words[:best_split])
    second_part = " ".join(words[best_split:])
    
    result = [first_part, second_part]
    logger.warning(f"🔧 强制分割完成: {' -- '.join(result)}")
    return result