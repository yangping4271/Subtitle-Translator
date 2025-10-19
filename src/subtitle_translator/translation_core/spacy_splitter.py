#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 spaCy 的智能句子分割模块
使用依存关系分析进行语法感知的句子分割
"""

import re
from typing import List, Tuple, Optional
from ..logger import setup_logger

logger = setup_logger("spacy_splitter")

# 全局变量，用于懒加载 spaCy 模型
_nlp = None
_spacy_available = None

def _is_spacy_available() -> bool:
    """检查 spaCy 是否可用"""
    global _spacy_available
    if _spacy_available is not None:
        return _spacy_available

    try:
        import spacy
        _spacy_available = True
        return True
    except ImportError:
        logger.info("spaCy 未安装，将跳过语法分析分割")
        _spacy_available = False
        return False

def _load_spacy_model():
    """懒加载 spaCy 模型"""
    global _nlp
    if _nlp is not None:
        return _nlp

    if not _is_spacy_available():
        return None

    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy 英文模型加载成功")
        return _nlp
    except OSError:
        logger.info("未找到 en_core_web_sm 模型，跳过 spaCy 分割")
        return None
    except Exception as e:
        logger.info(f"spaCy 模型加载失败: {e}")
        return None

def count_words(text: str) -> int:
    """统计英文单词数"""
    english_text = re.sub(r'[\u4e00-\u9fff]', ' ', text)
    english_words = english_text.strip().split()
    return len(english_words)

def find_split_points(doc) -> List[Tuple[int, str, str]]:
    """
    查找句子中的最佳分割点
    
    Args:
        doc: spaCy 处理后的文档对象
        
    Returns:
        分割点列表，每个元素是 (token索引, 词语, 分割原因)
    """
    split_points = []
    tokens = list(doc)
    
    for i, token in enumerate(tokens):
        # 跳过句首句尾附近的token
        if i < 2 or i >= len(tokens) - 2:
            continue
        
        # 1. 从属连词（引导状语从句）- 优先级最高
        if (token.dep_ == "mark" and 
            token.text.lower() in ["because", "since", "when", "where", "if", "unless", 
                                 "although", "though", "while", "whereas", "whether"]):
            split_points.append((i, token.text, "从属连词"))
        
        # 2. 并列连词 + 完整句子结构 - 需要验证前后都有动词
        elif (token.dep_ == "cc" and 
              token.text.lower() in ["and", "but", "or", "so", "yet"]):
            
            # 检查前后是否有动词结构
            has_verb_before = any(t.pos_ == "VERB" for t in tokens[:i] if i - t.i <= 10)
            has_verb_after = any(t.pos_ == "VERB" for t in tokens[i+1:] if t.i - i <= 10)
            
            if has_verb_before and has_verb_after:
                split_points.append((i, token.text, "并列连词+动词结构"))
        
        # 3. 并列动词（conj关系的动词）
        elif token.dep_ == "conj" and token.pos_ == "VERB":
            split_points.append((i, token.text, "并列动词"))
    
    # 按优先级排序：从属连词 > 并列连词+动词结构 > 并列动词
    priority_order = {"从属连词": 1, "并列连词+动词结构": 2, "并列动词": 3}
    split_points.sort(key=lambda x: (priority_order.get(x[2], 4), x[0]))
    
    return split_points

def validate_split(segments: List[str], min_words: int = 3) -> bool:
    """
    验证分割结果是否合理

    Args:
        segments: 分割后的句子片段
        min_words: 每个片段的最小词数

    Returns:
        True 如果分割合理
    """
    if len(segments) < 2:
        return False

    # 检查每个片段的长度
    for segment in segments:
        if count_words(segment.strip()) < min_words:
            return False

    # 检查长度平衡性（最长不超过最短的6倍,放宽限制以提高成功率）
    lengths = [count_words(seg.strip()) for seg in segments]
    if max(lengths) > min(lengths) * 6:
        return False

    return True

def spacy_split(text: str, max_segments: int = 3) -> List[str]:
    """
    使用 spaCy 进行基于语法分析的智能句子分割
    
    Args:
        text: 需要分割的句子
        max_segments: 最大分割片段数
        
    Returns:
        分割后的句子列表，如果分割失败返回空列表
    """
    # 加载模型
    nlp = _load_spacy_model()
    if nlp is None:
        return []
    
    try:
        # 分析句子结构
        doc = nlp(text)
        tokens = [token.text for token in doc]
        
        # 查找分割点
        split_points = find_split_points(doc)

        if not split_points:
            logger.info("未找到合适的语法分割点")
            return []

        # 选择最佳分割点（策略：只选1个最优点进行二分,避免切分过细）
        selected_points = []

        # 策略1: 优先选择从属连词（语义最强的分割点）
        subordinate_points = [p for p in split_points if p[2] == "从属连词"]
        if subordinate_points:
            # 选择最接近句子中点的从属连词
            tokens_list = list(doc)
            mid_pos = len(tokens_list) // 2
            best_point = min(subordinate_points, key=lambda p: abs(p[0] - mid_pos))
            selected_points.append(best_point)
            logger.info(f"选择从属连词分割点: {best_point}")

        # 策略2: 如果没有从属连词,选择并列连词
        elif len(split_points) > 0:
            conj_points = [p for p in split_points if p[2] == "并列连词+动词结构"]
            if conj_points:
                tokens_list = list(doc)
                mid_pos = len(tokens_list) // 2
                best_point = min(conj_points, key=lambda p: abs(p[0] - mid_pos))
                selected_points.append(best_point)
                logger.info(f"选择并列连词分割点: {best_point}")
            else:
                # 策略3: 选择并列动词
                verb_points = [p for p in split_points if p[2] == "并列动词"]
                if verb_points:
                    tokens_list = list(doc)
                    mid_pos = len(tokens_list) // 2
                    best_point = min(verb_points, key=lambda p: abs(p[0] - mid_pos))
                    selected_points.append(best_point)
                    logger.info(f"选择并列动词分割点: {best_point}")

        if not selected_points:
            logger.info("未选择任何分割点")
            return []
        
        # 执行分割
        segments = []
        start = 0
        
        for point_idx, word, reason in selected_points:
            segment = " ".join(tokens[start:point_idx]).strip()
            if segment:
                segments.append(segment)
            start = point_idx
        
        # 添加最后一段
        last_segment = " ".join(tokens[start:]).strip()
        if last_segment:
            segments.append(last_segment)
        
        # 验证分割结果
        if validate_split(segments):
            reason_info = ", ".join([f"{word}({reason})" for _, word, reason in selected_points])
            logger.info(f"✅ [策略1] spaCy语法分析分割在{reason_info}处: {len(segments)}段")
            return segments
        else:
            logger.info("spaCy分割结果验证失败")
            return []

    except Exception as e:
        logger.info(f"spaCy分割异常: {e}")
        return []

def get_spacy_info() -> str:
    """获取 spaCy 状态信息"""
    if not _is_spacy_available():
        return "spaCy未安装"
    
    nlp = _load_spacy_model()
    if nlp is None:
        return "spaCy可用，但en_core_web_sm模型未找到"
    
    return "spaCy就绪"