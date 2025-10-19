from concurrent.futures import ThreadPoolExecutor
import json
import re
from typing import Dict, Optional, List
import concurrent.futures

import retry
from openai import OpenAI

from .prompts import (
    TRANSLATE_PROMPT,
    SINGLE_TRANSLATE_PROMPT
)
from .config import SubtitleConfig
from .utils.json_repair import parse_llm_response
from ..logger import setup_logger

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
        config: Optional[SubtitleConfig] = None
    ):
        self.config = config or SubtitleConfig()
        self.client = OpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key
        )
        self.thread_num = self.config.thread_num
        self.batch_num = self.config.batch_size
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)
        # 使用列表存储日志
        self.batch_logs = []

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
            result = self.translate_multi_thread(subtitle_json, summary_content)

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

            # 检查翻译结果质量
            failed_count = 0
            for k, v in result["translated_subtitles"].items():
                if isinstance(v, str) and v.startswith("[翻译失败]"):
                    failed_count += 1
                elif isinstance(v, dict) and v.get("translation", "").startswith("[翻译失败]"):
                    failed_count += 1
            
            # 如果所有翻译都失败，抛出异常
            if failed_count == len(result["translated_subtitles"]):
                from .spliter import TranslationError
                suggestion = "💡 建议：请检查翻译模型名称是否正确，或更换其他可用模型"
                raise TranslationError("所有字幕翻译均失败", suggestion)
            
            # 如果部分翻译失败，记录警告
            if failed_count > 0:
                total_count = len(result["translated_subtitles"])
                logger.warning(f"⚠️ {failed_count}/{total_count} 条字幕翻译失败")
            
            # 转换结果格式
            translated_subtitle = []
            for k, v in result["optimized_subtitles"].items():
                translated_text = {
                    "id": int(k),
                    "original": subtitle_json[str(k)],
                    "optimized": v,
                    "translation": result["translated_subtitles"][k]
                }
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

    def translate_multi_thread(self, subtitle_json: Dict[int, str], summary_content: Dict = None):
        """多线程批量翻译字幕"""
        try:
            result, failed_chunks = self._batch_translate(subtitle_json, summary_content=summary_content)

            # 如果有失败的批次，使用单条翻译处理
            if failed_chunks:
                logger.info(f"有{len(failed_chunks)}个翻译批次失败，使用单条翻译处理这些批次")
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

    def _batch_translate(self, subtitle_json: Dict[int, str], summary_content: Dict = None) -> tuple[Dict, list]:
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
        logger.info(f"📋 翻译任务规划: {len(chunks)}个批次，每批次约{self.batch_num}条字幕")
        
        adjusted_count = getattr(self, '_adjusted_batch_count', 0)
        if adjusted_count > 0:
            logger.info(f"🔧 已优化{adjusted_count}个批次边界，确保句子完整性")
        
        # 检查是否达到最大线程限制
        actual_threads = min(len(chunks), self.thread_num)
        logger.info(f"⚡ 并发线程: {actual_threads}个")
        
        # 创建翻译任务
        futures = []
        chunk_map = {}  # 用于记录future和chunk的对应关系

        for i, chunk in enumerate(chunks):
            future = self.executor.submit(self._translate, chunk, summary_content, i+1, len(chunks))
            futures.append(future)
            chunk_map[future] = chunk
        
        # 收集结果
        optimized_subtitles = {}
        translated_subtitles = {}
        failed_chunks = []  # 记录失败的批次
        
        total = len(futures)
        completed = 0
        
        for future in concurrent.futures.as_completed(futures):
            completed += 1
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
                
                # 显示进度（每完成25%显示一次）
                progress_percentage = completed / total * 100
                if completed == 1 or completed == total or progress_percentage % 25 < (100 / total):
                    logger.info(f"🚀 翻译进度: {progress_percentage:.0f}% ({completed}/{total})")
                    
            except Exception as e:
                failed_chunk = chunk_map[future]
                logger.error(f"❌ 批次翻译失败: {e}")
                failed_chunks.append(failed_chunk)
        
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
                    model=self.config.translation_model,
                    stream=False,
                    messages=message,
                    temperature=0.7,
                    timeout=80
                    )
                message.pop()
                
                # 添加类型检查
                if isinstance(response, str):
                    logger.error(f"❌ API调用返回错误: {response}")
                    raise Exception(f"API调用失败: {response}")
                
                if not hasattr(response, 'choices') or not response.choices:
                    logger.error("❌ API响应格式异常：缺少choices属性")
                    raise Exception("API响应格式异常")
                
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
                                summary_content: Dict):
        """创建翻译提示消息"""
        # 基础输入内容 - 使用json.dumps确保格式正确
        input_content = (f"Correct and translate the following subtitles into {self.config.target_language}:\n"
                        f"<subtitles>{json.dumps(original_subtitle, ensure_ascii=False)}</subtitles>")
        
        # 解析并构建结构化的参考信息
        if summary_content and 'summary' in summary_content:
            try:
                # 解析总结JSON
                summary_json = parse_llm_response(summary_content.get('summary', '{}'))
                
                # 构建简洁的参考信息
                reference_parts = []
                
                # 添加上下文信息
                if context := summary_json.get('context'):
                    reference_parts.append(
                        f"Context: {context.get('type', '')} - {context.get('topic', '')}"
                    )
                
                # 添加纠错映射
                if corrections := summary_json.get('corrections'):
                    reference_parts.append(
                        f"Apply corrections: {json.dumps(corrections, ensure_ascii=False)}"
                    )
                
                # 添加不翻译列表
                if do_not_translate := summary_json.get('do_not_translate'):
                    reference_parts.append(
                        f"Keep in original: {', '.join(do_not_translate)}"
                    )
                
                # 添加规范术语
                if canonical := summary_json.get('canonical_terms'):
                    reference_parts.append(
                        f"Use canonical forms: {', '.join(canonical[:10])}"  # 限制显示前10个
                    )
                
                # 组合参考信息
                if reference_parts:
                    input_content += "\n\n<reference>\n" + "\n".join(reference_parts) + "\n</reference>"
                    
            except Exception as e:
                logger.warning(f"Failed to parse summary content: {e}")
                # 降级处理：使用原始方式
                input_content += (f"\n\nReference information:\n"
                                f"<reference>{summary_content.get('summary', '')}</reference>")

        prompt = TRANSLATE_PROMPT
        prompt = prompt.replace("[TargetLanguage]", self.config.target_language)

        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_content}
        ]

    def _print_all_batch_logs(self):
        """统一打印所有批次的日志"""
        if not self.batch_logs:
            return
            
        logger.info("📊 字幕优化结果汇总")

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

        # 统计变更类型
        format_changes = 0
        content_changes = 0
        wrong_changes = 0

        # 遍历所有日志，只打印有实际改动的
        change_count = 0
        for log in self.batch_logs:
            if log["type"] == "content_optimization":
                id_num = log["id"]
                original = log["original"]
                optimized = log["optimized"]

                # 只在实际有变化时打印
                if original != optimized:
                    change_count += 1
                    logger.info(f"🔧 字幕ID {id_num} - 内容优化:")
                    logger.info(f"   原文: {original}")
                    logger.info(f"   优化: {optimized}")

                    # 分类统计
                    if is_format_change_only(original, optimized):
                        format_changes += 1
                    elif is_wrong_replacement(original, optimized):
                        wrong_changes += 1
                    else:
                        content_changes += 1

        # 显示统计摘要
        logger.info("📈 优化统计:")
        logger.info(f"   格式优化: {format_changes} 项")
        logger.info(f"   内容修改: {content_changes} 项")
        if wrong_changes > 0:
            logger.info(f"   ⚠️ 可疑替换: {wrong_changes} 项")
        
        total_changes = format_changes + content_changes + wrong_changes
        logger.info(f"   总计修改: {total_changes} 项")
        logger.info("✅ 字幕优化汇总完成")

    @retry.retry(tries=2)
    def _translate(self, original_subtitle: Dict[str, str], 
                  summary_content: Dict, batch_num=None, total_batches=None) -> List[Dict]:
        """翻译字幕"""
        subtitle_keys = sorted(map(int, original_subtitle.keys()))
        batch_info = f"[批次{batch_num}/{total_batches}]" if batch_num and total_batches else ""
        
        logger.info(f"🌍 {batch_info} 翻译 {len(subtitle_keys)} 条字幕")

        max_retries = 2  # 最大重试次数
        current_try = 0
        
        while current_try < max_retries:
            try:
                message = self._create_translate_message(original_subtitle, summary_content)

                # 【关键日志】记录提交给LLM的原始输入数据
                logger.info(f"📤 {batch_info} 提交给LLM的字幕数据 (共{len(original_subtitle)}条):")
                logger.info(f"   输入JSON: {json.dumps(original_subtitle, ensure_ascii=False)}")

                response = self.client.chat.completions.create(
                    model=self.config.translation_model,
                    stream=False,
                    messages=message,
                    temperature=0.7,
                    timeout=80
                )
                # 添加类型检查
                if isinstance(response, str):
                    logger.error(f"❌ API调用返回错误: {response}")
                    raise Exception(f"API调用失败: {response}")
                
                if not hasattr(response, 'choices') or not response.choices:
                    logger.error("❌ API响应格式异常：缺少choices属性")
                    raise Exception("API响应格式异常")
                
                # 获取原始响应内容
                raw_response = response.choices[0].message.content
                logger.info(f"📥 {batch_info} LLM原始返回数据:\n{raw_response}")

                response_content = parse_llm_response(raw_response)
                logger.info(f"📥 {batch_info} 解析后的数据类型: {type(response_content)}")

                # 🔧 类型检查和自动修复
                if isinstance(response_content, list):
                    logger.warning(f"⚠️ {batch_info} LLM返回了array而非object，尝试转换")
                    logger.info(f"📊 {batch_info} Array内容: {json.dumps(response_content, ensure_ascii=False)}")
                    try:
                        # 尝试从list转换为dict
                        new_dict = {}
                        for item in response_content:
                            if isinstance(item, dict):
                                # 尝试多种可能的ID字段名
                                item_id = item.get('id') or item.get('subtitle_id') or item.get('key')
                                if item_id:
                                    new_dict[str(item_id)] = {
                                        'optimized_subtitle': item.get('optimized_subtitle', item.get('optimized', '')),
                                        'translation': item.get('translation', '')
                                    }
                        if new_dict:
                            response_content = new_dict
                            logger.info(f"✅ {batch_info} 成功转换array为object，包含{len(new_dict)}个条目")
                        else:
                            logger.error(f"❌ {batch_info} Array转换失败：无法提取有效数据")
                            logger.error(f"❌ {batch_info} 失败原因：array中没有可识别的id字段")
                            response_content = {}
                    except Exception as e:
                        logger.error(f"❌ {batch_info} Array转换异常: {e}")
                        logger.error(f"❌ {batch_info} Array结构: {json.dumps(response_content[:2] if len(response_content) > 2 else response_content, ensure_ascii=False)}")
                        response_content = {}

                if not isinstance(response_content, dict):
                    logger.error(f"❌ {batch_info} LLM返回类型错误: {type(response_content)}")
                    logger.error(f"❌ {batch_info} 返回内容: {str(response_content)[:500]}")
                    raise Exception(f"LLM返回格式错误，期望dict，实际{type(response_content)}")

                logger.info(f"📥 {batch_info} API返回结果样例（前3条）:")
                # 只显示前3条翻译结果作为样例
                sample_keys = list(response_content.keys())[:3] if response_content else []
                for k in sample_keys:
                    if k in response_content:
                        logger.info(f"   ID {k}: {response_content[k]}")

                # 如果完全没有返回结果，这是整批次的失败，需要重试
                if not response_content:
                    current_try += 1
                    if current_try < max_retries:
                        logger.warning(f"⚠️ {batch_info} API返回空结果，第{current_try}次重试")
                        continue
                    logger.error(f"❌ {batch_info} 重试{max_retries}次仍失败，使用默认翻译")
                    response_content = {}

                # 【关键日志】对比输入和返回的ID
                input_ids = set(original_subtitle.keys())
                output_ids = set(response_content.keys())
                missing_ids = input_ids - output_ids
                extra_ids = output_ids - input_ids

                if missing_ids:
                    logger.warning(f"⚠️ {batch_info} LLM丢失了这些ID: {sorted([int(x) for x in missing_ids])}")
                if extra_ids:
                    logger.warning(f"⚠️ {batch_info} LLM返回了额外的ID: {sorted([int(x) for x in extra_ids])}")

                # 检查API返回的结果是否完整
                problematic_ids = []
                for k in original_subtitle.keys():
                    if str(k) not in response_content:
                        logger.warning(f"⚠️ API返回结果缺少字幕ID: {k}")
                        logger.warning(f"⚠️ 原始字幕: {original_subtitle[str(k)]}")
                        problematic_ids.append(k)
                        response_content[str(k)] = {
                            "optimized_subtitle": original_subtitle[str(k)],
                            "translation": f"[翻译失败] {original_subtitle[str(k)]}"
                        }
                    elif "optimized_subtitle" not in response_content[str(k)]:
                        logger.warning(f"⚠️ 字幕ID {k} 缺少optimized_subtitle字段")
                        logger.warning(f"⚠️ 该字幕返回的数据: {json.dumps(response_content[str(k)], ensure_ascii=False)}")
                        response_content[str(k)]["optimized_subtitle"] = original_subtitle[str(k)]
                        problematic_ids.append(k)
                    elif "translation" not in response_content[str(k)]:
                        logger.warning(f"⚠️ 字幕ID {k} 缺少translation字段")
                        logger.warning(f"⚠️ 该字幕返回的数据: {json.dumps(response_content[str(k)], ensure_ascii=False)}")
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
                        self.batch_logs.append({
                            'type': 'content_optimization',
                            'id': k,
                            'original': translated_text['original'],
                            'optimized': translated_text['optimized']
                        })
                
                # 记录翻译示例（调试用）
                if translated_subtitle:
                    logger.info(f"✅ {batch_info} 翻译完成示例:")
                    for item in translated_subtitle[:2]:  # 显示前2个翻译结果
                        logger.info(f"   原文: {item['optimized']}")
                        logger.info(f"   译文: {item['translation']}")

                return translated_subtitle

            except Exception as e:
                current_try += 1
                if current_try < max_retries:
                    logger.error(f"❌ {batch_info} 翻译失败，第{current_try}次重试: {e}")
                    continue
                logger.error(f"❌ {batch_info} 重试{max_retries}次仍失败: {e}")
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