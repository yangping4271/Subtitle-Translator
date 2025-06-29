from concurrent.futures import ThreadPoolExecutor
import json
import re
from typing import Dict, Optional, List
import concurrent.futures

import retry
from openai import OpenAI

from .prompts import (
    TRANSLATE_PROMPT,
    REFLECT_TRANSLATE_PROMPT,
    SINGLE_TRANSLATE_PROMPT
)
from .config import SubtitleConfig
from .utils.json_repair import parse_llm_response
from .utils.logger import setup_logger

logger = setup_logger("subtitle_optimizer")

def is_sentence_complete(text: str) -> bool:
    """
    检查句子是否完整
    
    Args:
        text: 要检查的文本
        
    Returns:
        bool: 如果句子完整则返回True，否则返回False
    """
    # 句子结束标志
    sentence_end_markers = ['.', '!', '?', '。', '！', '？', '…']
    
    # 不应该结束于此的词语
    bad_end_words = ["and", "or", "but", "so", "yet", "for", "nor", "in", "on", "at", "to", "with", "by", "as"]
    
    # 检查是否以句子结束标志结尾
    text = text.strip()
    if not text:
        return True
        
    # 检查最后一个字符是否是句子结束标志
    if any(text.endswith(marker) for marker in sentence_end_markers):
        return True
        
    # 检查是否以不好的词结尾
    for word in bad_end_words:
        if text.lower().endswith(" " + word) or text.lower() == word:
            return False
            
    # 如果没有明确的结束标志，检查是否可能是不完整的句子
    words = text.split()
    if len(words) < 3:  # 如果句子太短，可能不完整
        return False
        
    return True

class SubtitleOptimizer:
    """A class for optimize and translating subtitles using OpenAI's API."""

    def __init__(
        self,
        config: Optional[SubtitleConfig] = None,
        need_reflect: bool = False
    ):
        self.config = config or SubtitleConfig()
        self.need_reflect = need_reflect
        self.client = OpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key
        )
        self.thread_num = self.config.thread_num
        self.batch_num = self.config.batch_size
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)
        # 改用字典存储日志，使用ID作为键以自动去重
        self.batch_logs = {}

    def translate(self, asr_data, summary_content: Dict) -> List[Dict]:
        """
        翻译字幕
        Args:
            asr_data: ASR识别结果
            summary_content: 总结内容，包含summary和readable_name
        Returns:
            List[Dict]: 翻译结果列表
        """
        try:
            # 清空之前的日志
            self.batch_logs.clear()
            
            subtitle_json = {str(k): v["original_subtitle"] 
                            for k, v in asr_data.to_json().items()}
            
            # 使用多线程批量翻译
            result = self.translate_multi_thread(subtitle_json, self.need_reflect, summary_content)

            # 检查是否有翻译失败的字幕（带有[翻译失败]前缀）
            failed_subtitles = {}
            for k, v in result["translated_subtitles"].items():
                if isinstance(v, str) and v.startswith("[翻译失败]"):
                    failed_subtitles[k] = subtitle_json[k]
                elif isinstance(v, dict) and v.get("translation", "").startswith("[翻译失败]"):
                    failed_subtitles[k] = subtitle_json[k]
            
            # 如果有翻译失败的字幕，使用单条翻译再次尝试
            if failed_subtitles:
                logger.info(f"发现{len(failed_subtitles)}个字幕翻译失败，使用单条翻译再次尝试")
                retry_result = self._translate_chunk_by_single(failed_subtitles)
                
                # 更新结果
                for k, v in retry_result["translated_subtitles"].items():
                    if not v.startswith("[翻译失败]"):
                        logger.info(f"字幕ID {k} 单条翻译成功")
                        result["optimized_subtitles"][str(k)] = retry_result["optimized_subtitles"][k]
                        result["translated_subtitles"][str(k)] = v

            # 转换结果格式
            translated_subtitle = []
            for k, v in result["optimized_subtitles"].items():
                translated_text = {
                    "id": int(k),
                    "original": subtitle_json[str(k)],
                    "optimized": v,
                    "translation": result["translated_subtitles"][k]
                }
                # 如果是反思模式，添加反思相关的字段
                if self.need_reflect and isinstance(result["translated_subtitles"][k], dict):
                    translated_text.update({
                        "revised_translation": result["translated_subtitles"][k].get("revised_translation"),
                        "revise_suggestions": result["translated_subtitles"][k].get("revise_suggestions"),
                        "translation": result["translated_subtitles"][k].get("translation")
                    })
                translated_subtitle.append(translated_text)
            
            # logger.info(f"翻译结果: {json.dumps(translated_subtitle, indent=4, ensure_ascii=False)}")
            
            # 所有批次处理完成后，统一输出日志
            self._print_all_batch_logs()
            return translated_subtitle
        finally:
            self.stop()  # 确保线程池被关闭

    def stop(self):
        """优雅关闭线程池"""
        if hasattr(self, 'executor'):
            try:
                logger.info("正在等待线程池任务完成...")
                self.executor.shutdown(wait=True)
                logger.info("线程池已关闭")
            except Exception as e:
                logger.error(f"关闭线程池时发生错误: {e}")
            finally:
                self.executor = None

    def translate_multi_thread(self, subtitle_json: Dict[int, str], reflect: bool = False, 
                             summary_content: Dict = None):
        """多线程批量翻译字幕"""
        if reflect:
            try:
                result, failed_chunks = self._batch_translate(subtitle_json, use_reflect=True, summary_content=summary_content)
                
                # 如果有失败的批次，使用单条翻译处理
                if failed_chunks:
                    logger.info(f"有{len(failed_chunks)}个反思翻译批次失败，使用单条翻译处理这些批次")
                    # 将失败的批次合并成一个字典
                    failed_subtitles = {}
                    for chunk in failed_chunks:
                        failed_subtitles.update(chunk)
                    
                    # 只对失败的字幕使用单条翻译
                    single_result = self._translate_by_single(failed_subtitles)
                    
                    # 合并结果
                    result["optimized_subtitles"].update(single_result["optimized_subtitles"])
                    result["translated_subtitles"].update(single_result["translated_subtitles"])
                
                return result
            except Exception as e:
                logger.error(f"反思翻译完全失败，使用单条翻译处理所有内容：{e}")
                return self._translate_by_single(subtitle_json)
        
        try:
            # 尝试批量翻译
            result, failed_chunks = self._batch_translate(subtitle_json, use_reflect=False, summary_content=summary_content)
            
            # 如果有失败的批次，使用单条翻译处理
            if failed_chunks:
                logger.info(f"有{len(failed_chunks)}个批次翻译失败，使用单条翻译处理这些批次")
                # 将失败的批次合并成一个字典
                failed_subtitles = {}
                for chunk in failed_chunks:
                    failed_subtitles.update(chunk)
                
                # 只对失败的字幕使用单条翻译
                single_result = self._translate_by_single(failed_subtitles)
                
                # 合并结果
                result["optimized_subtitles"].update(single_result["optimized_subtitles"])
                result["translated_subtitles"].update(single_result["translated_subtitles"])
            
            return result
        except Exception as e:
            logger.error(f"批量翻译完全失败，使用单条翻译处理所有内容：{e}")
            return self._translate_by_single(subtitle_json)

    def _batch_translate(self, subtitle_json: Dict[int, str], use_reflect: bool = False, 
                         summary_content: Dict = None) -> tuple[Dict, list]:
        """批量翻译字幕的核心方法
        
        Returns:
            tuple: (翻译结果字典, 失败批次列表)
        """
        items = list(subtitle_json.items())[:]
        
        # 修改批次切分逻辑，确保每个批次的最后一句是完整的
        chunks = []
        i = 0
        self._adjusted_batch_count = 0  # 初始化调整计数器
        
        while i < len(items):
            # 确定当前批次的结束位置
            end_idx = min(i + self.batch_num, len(items))
            
            # 如果不是最后一个批次，检查最后一句是否完整
            if end_idx < len(items):
                # 获取当前批次的最后一句
                last_id, last_text = items[end_idx - 1]
                
                # 检查最后一句是否完整
                if not is_sentence_complete(last_text):
                    logger.info(f"批次结束于不完整句子: '{last_text}'，尝试调整批次边界")
                    self._adjusted_batch_count += 1  # 增加调整计数器
                    
                    # 向前查找完整句子的位置
                    complete_idx = end_idx - 1
                    while complete_idx > i and not is_sentence_complete(items[complete_idx - 1][1]):
                        complete_idx -= 1
                    
                    # 如果找到了完整句子，调整批次边界
                    if complete_idx > i:
                        logger.info(f"调整批次边界: {end_idx} -> {complete_idx} (确保句子完整性)")
                        end_idx = complete_idx
                    else:
                        # 如果向前找不到完整句子，尝试向后查找
                        complete_idx = end_idx
                        while complete_idx < len(items) and not is_sentence_complete(items[complete_idx - 1][1]):
                            complete_idx += 1
                            
                            # 设置一个合理的向后查找限制，避免批次过大
                            if complete_idx - i > self.batch_num * 1.5:
                                break
                        
                        if complete_idx < len(items):
                            logger.info(f"调整批次边界: {end_idx} -> {complete_idx} (确保句子完整性)")
                            end_idx = complete_idx
                        else:
                            logger.warning(f"无法找到完整句子边界，使用原始批次边界: {end_idx}")
            
            # 创建当前批次
            chunk = dict(items[i:end_idx])
            chunks.append(chunk)
            
            # 更新起始位置
            i = end_idx
        
        # 记录批次信息
        logger.info(f"开始批量翻译任务: 预设每批次{self.batch_num}条字幕")
        logger.info(f"共{len(chunks)}个批次, 平均{sum(len(chunk) for chunk in chunks)/len(chunks):.0f}条字幕")
        
        adjusted_count = getattr(self, '_adjusted_batch_count', 0)
        if adjusted_count > 0:
            logger.info(f"有{adjusted_count}个批次因句子不完整而进行了调整，确保句子完整性")
        
        # 检查是否达到最大线程限制
        actual_threads = min(len(chunks), self.thread_num)
        if actual_threads < self.thread_num:
            logger.info(f"实际使用线程数: {actual_threads}/{self.thread_num}")
        else:
            logger.info(f"实际使用线程数: {actual_threads}/{self.thread_num} (已达到配置的最大线程数)")
        
        # 创建翻译任务
        futures = []
        chunk_map = {}  # 用于记录future和chunk的对应关系
        
        for i, chunk in enumerate(chunks):
            if use_reflect:
                future = self.executor.submit(self._reflect_translate, chunk, summary_content, i+1, len(chunks))
            else:
                future = self.executor.submit(self._translate, chunk, summary_content, i+1, len(chunks))
            futures.append(future)
            chunk_map[future] = chunk
        
        # 收集结果
        optimized_subtitles = {}
        translated_subtitles = {}
        failed_chunks = []  # 记录失败的批次
        
        total = len(futures)
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                result = future.result()
                for item in result:
                    k = str(item["id"])
                    optimized_subtitles[k] = item["optimized"]
                    # 保存完整的翻译信息
                    if "revised_translation" in item:
                        translated_subtitles[k] = {
                            "translation": item["translation"],
                            "revised_translation": item["revised_translation"],
                            "revise_suggestions": item["revise_suggestions"]
                        }
                    else:
                        translated_subtitles[k] = item["translation"]
                logger.info(f"批量翻译进度: 第{i}/{total} 已完成翻译")
            except Exception as e:
                logger.error(f"批量翻译任务失败（批次 {i}/{total}）：{e}")
                # 记录失败的批次，而不是立即抛出异常
                failed_chunks.append(chunk_map[future])
        
        # 返回成功的结果和失败的批次
        return {
            "optimized_subtitles": optimized_subtitles,
            "translated_subtitles": translated_subtitles
        }, failed_chunks

    def _translate_by_single(self, subtitle_json: Dict[int, str]) -> Dict:
        """使用单条翻译模式处理字幕"""
        items = list(subtitle_json.items())[:]
        chunks = [dict(items[i:i + self.batch_num]) 
                 for i in range(0, len(items), self.batch_num)]
        
        # 创建翻译任务
        futures = []
        chunk_map = {}  # 用于记录future和chunk的对应关系
        
        for i, chunk in enumerate(chunks):
            future = self.executor.submit(self._translate_chunk_by_single, chunk)
            futures.append(future)
            chunk_map[future] = chunk
        
        # 收集结果
        optimized_subtitles = {}
        translated_subtitles = {}
        total = len(futures)
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                result = future.result()
                for k, v in result["optimized_subtitles"].items():
                    optimized_subtitles[str(k)] = v
                    translated_subtitles[str(k)] = result["translated_subtitles"][k]
                logger.info(f"单条翻译进度: 第{i}批次/{total}批次 已完成翻译")
            except Exception as e:
                logger.error(f"单条翻译任务失败（批次 {i}/{total}）：{e}")
                # 处理失败的批次，使用默认翻译
                failed_chunk = chunk_map[future]
                for k, v in failed_chunk.items():
                    optimized_subtitles[str(k)] = v
                    translated_subtitles[str(k)] = f"[翻译失败] {v}"
                logger.warning(f"已为失败的批次 {i}/{total} 使用默认翻译")
        
        return {
            "optimized_subtitles": optimized_subtitles,
            "translated_subtitles": translated_subtitles
        }

    @retry.retry(tries=2)
    def _translate_chunk_by_single(self, subtitle_chunk: Dict[int, str]) -> Dict:
        """单条翻译模式的核心方法"""
        subtitle_keys = sorted(map(int, subtitle_chunk.keys()))
        # 修改日志输出，只打印字幕数量而不是范围
        logger.info(f"[+]正在单条翻译字幕，共{len(subtitle_keys)}条")
        
        translated_subtitle = {}
        message = [{"role": "system",
                   "content": SINGLE_TRANSLATE_PROMPT.replace("[TargetLanguage]", self.config.target_language)}]
        
        for key, value in subtitle_chunk.items():
            try:
                # 为每个字幕ID添加单独的日志
                logger.info(f"[+]正在翻译字幕ID: {key}")
                message.append({"role": "user", "content": value})
                response = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    stream=False,
                    messages=message,
                    temperature=0.7,
                    timeout=80
                    )
                message.pop()
                
                translate = response.choices[0].message.content.strip()
                translated_subtitle[key] = translate
                logger.info(f"单条翻译原文: {value}")
                logger.info(f"单条翻译结果: {translate}")
            except Exception as e:
                logger.error(f"单条翻译失败，字幕ID: {key}，错误: {e}")
                # 使用默认翻译，而不是空字符串，这样用户至少能看到原文
                translated_subtitle[key] = f"[翻译失败] {value}"
        
        # 确保所有字幕都有翻译结果
        for key in subtitle_chunk.keys():
            if key not in translated_subtitle:
                logger.warning(f"字幕ID {key} 没有翻译结果，使用默认翻译")
                translated_subtitle[key] = f"[翻译失败] {subtitle_chunk[key]}"
        
        return {
            "optimized_subtitles": subtitle_chunk,
            "translated_subtitles": translated_subtitle
        }

    def _create_translate_message(self, original_subtitle: Dict[str, str], 
                                summary_content: Dict, reflect=False):
        """创建翻译提示消息"""
        input_content = (f"correct the original subtitles, and translate them into {self.config.target_language}:"
                        f"\n<input_subtitle>{str(original_subtitle)}</input_subtitle>")

        if summary_content:
            input_content += (f"\nThe following is reference material related to subtitles, based on which "
                            f"the subtitles will be corrected, optimized, and translated:"
                            f"\n<prompt>{summary_content.get('summary', '')}</prompt>\n")

        prompt = REFLECT_TRANSLATE_PROMPT if reflect else TRANSLATE_PROMPT
        prompt = prompt.replace("[TargetLanguage]", self.config.target_language)

        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_content}
        ]

    def _print_all_batch_logs(self):
        """统一打印所有批次的日志"""
        if not self.batch_logs:
            return
            
        logger.info("================ 字幕优化结果汇总 ================")

        def is_format_change_only(original, optimized):
            """判断是否只有格式变化（大小写和标点符号）"""
            import string
            # 忽略大小写和标点符号后比较
            original_normalized = original.lower().translate(str.maketrans('', '', string.punctuation))
            optimized_normalized = optimized.lower().translate(str.maketrans('', '', string.punctuation))
            return original_normalized == optimized_normalized

        def is_wrong_replacement(original, optimized):
            """检测是否存在错误的替换（替换了不相关的词）"""
            import re
            # 提取所有单词
            original_words = set(re.findall(r'\b\w+\b', original.lower()))
            optimized_words = set(re.findall(r'\b\w+\b', optimized.lower()))
            # 找出被替换的词
            removed_words = original_words - optimized_words
            added_words = optimized_words - original_words
            # 如果替换前后的词没有相似性，可能是错误替换
            if removed_words and added_words:
                for removed in removed_words:
                    for added in added_words:
                        # 如果原词和新词完全不同（编辑距离过大），判定为错误替换
                        if len(removed) > 3 and len(added) > 3 and not any(c in removed for c in added):
                            return True
            return False
            
        # 统计计数
        format_changes = 0
        content_changes = 0
        wrong_changes = 0
            
        # 按ID排序输出
        sorted_ids = sorted(self.batch_logs.keys())
        for i, id_num in enumerate(sorted_ids):
            log = self.batch_logs[id_num]
            original = log['original']
            optimized = log['optimized']
            
            # 判断改动类型并使用不同级别输出日志
            if is_format_change_only(original, optimized):
                format_changes += 1
                logger.debug(f"字幕ID {id_num} - 格式优化:")
                logger.debug(f"原始: {original}")
                logger.debug(f"优化: {optimized}")
                # 格式优化使用debug级别分隔线
                if i < len(sorted_ids) - 1:
                    logger.debug("-" * 50)
            else:
                if is_wrong_replacement(original, optimized):
                    wrong_changes += 1
                    logger.error(f"字幕ID {id_num} - 可能存在错误替换:")
                    logger.error(f"原始: {original}")
                    logger.error(f"优化: {optimized}")
                    # 错误替换使用error级别分隔线
                    if i < len(sorted_ids) - 1:
                        logger.error("-" * 50)
                else:
                    content_changes += 1
                    logger.info(f"字幕ID {id_num} - 内容优化:")
                    logger.info(f"原始: {original}")
                    logger.info(f"优化: {optimized}")
                    # 内容优化使用info级别分隔线
                    if i < len(sorted_ids) - 1:
                        logger.info("-" * 50)

            if 'revised_translation' in log and log['revised_translation'] != log['translation']:
                logger.info(f"字幕ID: {id_num} - 翻译优化:")
                logger.info(f"字幕: {log['optimized']}")
                logger.info(f"翻译: {log['translation']}")
                logger.info(f"反思建议: {log['revise_suggestions']}")
                logger.info(f"反思后翻译: {log['revised_translation']}")
                if i < len(sorted_ids) - 1:
                    logger.info("-" * 50)
        
        # 输出统计信息
        logger.info("统计信息:")
        logger.info(f"格式优化数量: {format_changes}")
        logger.info(f"内容修改数量: {content_changes}")
        if wrong_changes > 0:
            logger.error(f"疑似错误替换数量: {wrong_changes}")
        logger.info(f"总修改数量: {format_changes + content_changes + wrong_changes}")
        logger.info("================ 字幕优化结果结束 ================")
        # 清空日志字典
        self.batch_logs.clear()

    @retry.retry(tries=2)
    def _reflect_translate(self, original_subtitle: Dict[str, str], 
                          summary_content: Dict, batch_num=None, total_batches=None) -> List[Dict]:
        """反思翻译字幕"""
        subtitle_keys = sorted(map(int, original_subtitle.keys()))
        batch_info = f"[批次 {batch_num}/{total_batches}] " if batch_num and total_batches else ""
        if len(subtitle_keys) == self.batch_num:
            logger.info(f"[+]{batch_info}正在反思翻译字幕：{subtitle_keys[0]} - {subtitle_keys[-1]}")
        else:
            logger.info(f"[+]{batch_info}正在反思翻译字幕：{subtitle_keys[0]} - {subtitle_keys[-1]} (共{len(subtitle_keys)}条)")

        max_retries = 2  # 最大重试次数
        current_try = 0
        
        while current_try < max_retries:
            try:
                message = self._create_translate_message(original_subtitle, summary_content, reflect=True)
                response = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    stream=False,
                    messages=message,
                    temperature=0.7,
                    timeout=80
                )
                response_content = parse_llm_response(response.choices[0].message.content)
                
                logger.debug(f"反思翻译API返回结果: {json.dumps(response_content, indent=4, ensure_ascii=False)}")

                # 如果完全没有返回结果，这是整批次的失败，需要重试
                if not response_content:
                    current_try += 1
                    if current_try < max_retries:
                        logger.warning(f"反思翻译API返回空结果，第{current_try}次重试整个批次")
                        continue
                    logger.error(f"反思翻译批次重试{max_retries}次后仍然失败，将使用默认翻译")
                    response_content = {}

                # 检查API返回的结果是否完整
                problematic_ids = []
                for k in original_subtitle.keys():
                    if str(k) not in response_content:
                        logger.warning(f"API返回结果缺少字幕ID: {k}，将使用原始字幕")
                        problematic_ids.append(k)
                        response_content[str(k)] = {
                            "optimized_subtitle": original_subtitle[str(k)],
                            "translation": f"[翻译失败] {original_subtitle[str(k)]}",
                            "revised_translation": f"[翻译失败] {original_subtitle[str(k)]}",
                            "revise_suggestions": "翻译失败，无法提供反思建议"
                        }
                    else:
                        # 检查必要的字段是否存在
                        if "optimized_subtitle" not in response_content[str(k)]:
                            logger.warning(f"字幕ID {k} 缺少optimized_subtitle字段，将使用原始字幕")
                            response_content[str(k)]["optimized_subtitle"] = original_subtitle[str(k)]
                            problematic_ids.append(k)
                        
                        if "translation" not in response_content[str(k)]:
                            logger.warning(f"字幕ID {k} 缺少translation字段，将使用默认翻译")
                            response_content[str(k)]["translation"] = f"[翻译失败] {original_subtitle[str(k)]}"
                            problematic_ids.append(k)
                        
                        if "revised_translation" not in response_content[str(k)]:
                            logger.warning(f"字幕ID {k} 缺少revised_translation字段，将使用translation字段")
                            response_content[str(k)]["revised_translation"] = response_content[str(k)].get("translation", f"[翻译失败] {original_subtitle[str(k)]}")
                            problematic_ids.append(k)
                        
                        if "revise_suggestions" not in response_content[str(k)]:
                            logger.warning(f"字幕ID {k} 缺少revise_suggestions字段，将使用默认建议")
                            response_content[str(k)]["revise_suggestions"] = "翻译失败，无法提供反思建议"
                            problematic_ids.append(k)

                translated_subtitle = []
                for k, v in response_content.items():
                    k = int(k)
                    translated_text = {
                        "id": k,
                        "original": original_subtitle[str(k)],
                        "optimized": v["optimized_subtitle"],
                        "translation": v["translation"],
                        "revised_translation": v["revised_translation"],
                        "revise_suggestions": v["revise_suggestions"]
                    }
                    translated_subtitle.append(translated_text)

                    # 收集日志
                    if (translated_text["original"] != translated_text["optimized"] or 
                        translated_text["translation"] != translated_text["revised_translation"]):
                        self.batch_logs[k] = {
                            'original': translated_text['original'],
                            'optimized': translated_text['optimized'],
                            'translation': translated_text['translation'],
                            'revised_translation': translated_text['revised_translation'],
                            'revise_suggestions': translated_text['revise_suggestions']
                        }

                return translated_subtitle

            except Exception as e:
                current_try += 1
                if current_try < max_retries:
                    logger.error(f"反思翻译失败，第{current_try}次重试整个批次。错误：{e}")
                    continue
                logger.error(f"反思翻译失败，重试{max_retries}次后仍然失败。错误：{e}")
                # 创建默认的翻译结果
                translated_subtitle = []
                for k, v in original_subtitle.items():
                    k_int = int(k)
                    translated_text = {
                        "id": k_int,
                        "original": v,
                        "optimized": v,
                        "translation": f"[翻译失败] {v}",
                        "revised_translation": f"[翻译失败] {v}",
                        "revise_suggestions": "翻译失败，无法提供反思建议"
                    }
                    translated_subtitle.append(translated_text)
                return translated_subtitle

    @retry.retry(tries=2)
    def _translate(self, original_subtitle: Dict[str, str], 
                  summary_content: Dict, batch_num=None, total_batches=None) -> List[Dict]:
        """翻译字幕"""
        subtitle_keys = sorted(map(int, original_subtitle.keys()))
        batch_info = f"[批次 {batch_num}/{total_batches}] " if batch_num and total_batches else ""
        if len(subtitle_keys) == self.batch_num:
            logger.info(f"[+]{batch_info}正在翻译字幕：{subtitle_keys[0]} - {subtitle_keys[-1]}")
        else:
            logger.info(f"[+]{batch_info}正在翻译字幕：{subtitle_keys[0]} - {subtitle_keys[-1]} (共{len(subtitle_keys)}条)")

        max_retries = 2  # 最大重试次数
        current_try = 0
        
        while current_try < max_retries:
            try:
                message = self._create_translate_message(original_subtitle, summary_content, reflect=False)
                response = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    stream=False,
                    messages=message,
                    temperature=0.7,
                    timeout=80
                )
                response_content = parse_llm_response(response.choices[0].message.content)

                logger.debug(f"API返回结果: \n{json.dumps(response_content, indent=4, ensure_ascii=False)}\n")

                # 如果完全没有返回结果，这是整批次的失败，需要重试
                if not response_content:
                    current_try += 1
                    if current_try < max_retries:
                        logger.warning(f"API返回空结果，第{current_try}次重试整个批次")
                        continue
                    logger.error(f"批次重试{max_retries}次后仍然失败，将使用默认翻译")
                    response_content = {}

                # 检查API返回的结果是否完整
                problematic_ids = []
                for k in original_subtitle.keys():
                    if str(k) not in response_content:
                        logger.warning(f"API返回结果缺少字幕ID: {k}，将使用原始字幕")
                        problematic_ids.append(k)
                        response_content[str(k)] = {
                            "optimized_subtitle": original_subtitle[str(k)],
                            "translation": f"[翻译失败] {original_subtitle[str(k)]}"
                        }
                    elif "optimized_subtitle" not in response_content[str(k)]:
                        logger.warning(f"字幕ID {k} 缺少optimized_subtitle字段，将使用原始字幕")
                        response_content[str(k)]["optimized_subtitle"] = original_subtitle[str(k)]
                        problematic_ids.append(k)
                    elif "translation" not in response_content[str(k)]:
                        logger.warning(f"字幕ID {k} 缺少translation字段，将使用默认翻译")
                        response_content[str(k)]["translation"] = f"[翻译失败] {original_subtitle[str(k)]}"
                        problematic_ids.append(k)

                translated_subtitle = []
                for k, v in response_content.items():
                    k = int(k)
                    translated_text = {
                        "id": k,
                        "original": original_subtitle[str(k)],
                        "optimized": v["optimized_subtitle"],
                        "translation": v["translation"]
                    }
                    translated_subtitle.append(translated_text)

                    # 收集日志
                    if translated_text["original"] != translated_text["optimized"]:
                        self.batch_logs[k] = {
                            'original': translated_text['original'],
                            'optimized': translated_text['optimized'],
                            'translation': translated_text['translation']
                        }

                return translated_subtitle

            except Exception as e:
                current_try += 1
                if current_try < max_retries:
                    logger.error(f"翻译失败，第{current_try}次重试整个批次。错误：{e}")
                    continue
                logger.error(f"翻译失败，重试{max_retries}次后仍然失败。错误：{e}")
                # 创建默认的翻译结果
                translated_subtitle = []
                for k, v in original_subtitle.items():
                    k_int = int(k)
                    translated_text = {
                        "id": k_int,
                        "original": v,
                        "optimized": v,
                        "translation": f"[翻译失败] {v}"
                    }
                    translated_subtitle.append(translated_text)
                return translated_subtitle