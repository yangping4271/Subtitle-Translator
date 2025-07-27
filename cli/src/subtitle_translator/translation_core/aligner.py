import difflib
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class SubtitleAligner:
    """
    字幕文本对齐器，用于对齐两个文本序列。
    主要用于处理字幕优化后的文本对齐，确保字幕的顺序和对应关系正确。
    """
    # 相似度阈值常量
    SIMILARITY_THRESHOLD = 0.4  # 基本相似度阈值
    MINIMUM_ALIGNMENT_RATIO = 0.6  # 最小对齐比例

    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """
        计算两个文本的相似度。

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            float: 相似度（0-1之间的浮点数）
        """
        # 处理空字符串情况
        if not text1.strip() or not text2.strip():
            return 0.0
        
        # 基础相似度
        base_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        
        # 考虑长度差异
        len1, len2 = len(text1), len(text2)
        max_len = max(len1, len2)
        min_len = min(len1, len2)
        if max_len == 0:
            return 1.0  # 两个都是空字符串
        
        length_ratio = min_len / max_len
        
        # 综合考虑文本相似度和长度比例
        final_similarity = base_similarity * (0.7 + 0.3 * length_ratio)
        
        return final_similarity

    def align_texts(self, source_text: List[str], target_text: List[str]) -> Tuple[List[str], List[str]]:
        """
        对齐两个文本序列。纯粹的文本对齐功能，不处理字幕特定的逻辑。

        Args:
            source_text (List[str]): 源文本行列表
            target_text (List[str]): 目标文本行列表

        Returns:
            Tuple[List[str], List[str]]: 对齐后的源文本和目标文本

        Raises:
            ValueError: 当输入参数无效时
        """
        # 输入验证
        if not isinstance(source_text, list) or not isinstance(target_text, list):
            raise ValueError("输入必须是字符串列表")
        if not source_text or not target_text:
            raise ValueError("输入文本不能为空")
        
        # 计算文本相似度矩阵
        similarity_matrix = []
        for src in source_text:
            row = []
            for tgt in target_text:
                similarity = self._calculate_similarity(src, tgt)
                row.append(similarity)
            similarity_matrix.append(row)
        
        # 基于相似度进行对齐
        aligned_source = []
        aligned_target = []
        used_targets = set()
        
        for i, src in enumerate(source_text):
            best_match_idx = -1
            best_similarity = self.SIMILARITY_THRESHOLD
            
            # 找到最佳匹配
            for j, tgt in enumerate(target_text):
                if j not in used_targets and similarity_matrix[i][j] > best_similarity:
                    best_similarity = similarity_matrix[i][j]
                    best_match_idx = j
            
            # 添加对齐结果
            aligned_source.append(src)
            if best_match_idx != -1:
                aligned_target.append(target_text[best_match_idx])
                used_targets.add(best_match_idx)
            else:
                aligned_target.append("")  # 没有找到匹配时用空字符串
        
        # 计算整体相似度
        similarities = []
        for s, t in zip(aligned_source, aligned_target):
            if s and t:  # 忽略空行
                similarity = self._calculate_similarity(s, t)
                similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        logger.info(f"文本对齐完成，平均相似度: {avg_similarity:.2f}")
        
        return aligned_source, aligned_target

    def repair_subtitle(self, original_subtitle: Dict[int, str], optimized_subtitle: Dict[str, str]) -> Dict[int, str]:
        """
        修复字幕对齐问题，确保优化后的字幕与原字幕能够正确对应。
        处理字幕特定的逻辑，包括字典键的映射、空行处理和缺失行替换。

        Args:
            original_subtitle (Dict[int, str]): 原始字幕字典
            optimized_subtitle (Dict[str, str]): 优化后的字幕字典

        Returns:
            Dict[int, str]: 修复后的字幕字典

        Raises:
            ValueError: 当修复失败时抛出异常
        """
        # 1. 提取文本列表
        list1 = list(original_subtitle.values())
        list2 = list(optimized_subtitle.values())
        
        try:
            # 2. 进行文本对齐
            aligned_source, aligned_target = self.align_texts(list1, list2)
            
            # 3. 构建结果字典
            start_id = next(iter(original_subtitle.keys()))
            result = {}
            
            # 4. 处理每一行字幕
            for i, (src, tgt) in enumerate(zip(aligned_source, aligned_target)):
                current_id = int(start_id) + i
                
                if not tgt.strip():  # 如果目标文本为空，使用源文本
                    result[current_id] = src
                else:  # 否则使用目标文本
                    # 计算相似度，检查是否需要使用原文
                    similarity = self._calculate_similarity(src, tgt)
                    if similarity < self.SIMILARITY_THRESHOLD:  # 如果相似度过低，使用原文
                        result[current_id] = src
                        logger.warning(f"行 {current_id} 相似度过低 ({similarity:.2f})，使用原文")
                    else:
                        result[current_id] = tgt
            
            # 5. 检查是否所有原始字幕都有对应的结果
            for k in original_subtitle.keys():
                if k not in result:
                    result[k] = original_subtitle[k]
                    logger.warning(f"行 {k} 缺失，使用原文")
            
            logger.info("字幕对齐成功！")
            return result
            
        except Exception as e:
            logger.error(f"字幕对齐失败：{e}")
            raise ValueError(f"字幕对齐失败：{e}")

if __name__ == '__main__':
    # 使用示例
    text1 = ['Hello world', 'This is line 2', 'This is line 3']
    text2 = ['Hello world', '', 'This is line 3']
    
    aligner = SubtitleAligner()
    aligned_source, aligned_target = aligner.align_texts(text1, text2)
    
    print("对齐结果:")
    for s, t in zip(aligned_source, aligned_target):
        print(f"源文本: {s}")
        print(f"目标文本: {t}")
        print("---")
