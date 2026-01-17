import re
from typing import Dict, Optional
from pathlib import Path
from openai import OpenAI
from .prompts import SUMMARIZER_PROMPT
from .config import SubtitleConfig
from .utils.json_repair import parse_llm_response
from .utils.errors import extract_error_message, get_error_suggestions
from .utils.api import validate_api_response
from ..logger import setup_logger

logger = setup_logger("subtitle_summarizer")


class SubtitleSummarizer:
    def __init__(
        self,
        config: Optional[SubtitleConfig] = None
    ):
        self.config = config or SubtitleConfig()
        self.client = OpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key
        )

    def summarize(self, subtitle_content: str, input_file: str) -> Dict:
        """
        总结字幕内容
        Args:
            subtitle_content: 字幕内容
            input_file: 输入的字幕文件路径
        Returns:
            Dict: 包含总结信息的字典
        """
        try:
            # 使用 pathlib 处理文件名和路径
            path = Path(input_file)
            
            # 获取不带扩展名的文件名
            readable_filename = path.stem.replace('_', ' ').replace('-', ' ')
            
            # 提取文件夹路径信息 - 获取最后几级父目录
            parent_names = []
            current_path = path.parent
            # 最多获取3级父目录，避免过长的路径
            for i in range(3):
                if current_path.name and current_path.name not in ['/', '.', '..']:
                    folder_name = current_path.name.replace('_', ' ').replace('-', ' ')
                    parent_names.append(folder_name)
                    current_path = current_path.parent
                else:
                    break
            
            # 构建上下文信息
            context_parts = []
            if parent_names:
                folder_path_str = ' / '.join(reversed(parent_names))
                context_parts.append(f"Folder path: {folder_path_str}")
                
            context_parts.append(f"Filename: {readable_filename}")
            context_info = "\n".join(context_parts)

            logger.info(f"📋 可读性文件名: {readable_filename}")
            if parent_names:
                logger.info(f"📂 文件夹路径: {' / '.join(reversed(parent_names))}")
            
            # 更新提示词，强调文件名和路径的权威性
            from datetime import datetime
            current_date = datetime.now().strftime('%Y-%m-%d')

            message = [
                {
                    "role": "system",
                    "content": SUMMARIZER_PROMPT.format(current_date=current_date)
                },
                {"role": "user", "content": f"{context_info}\n\nContent:\n{subtitle_content}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.summary_model,
                messages=message,
                temperature=0.7,
                timeout=80
            )

            summary = validate_api_response(response)

            # 移除<think>和</think>标签
            summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)

            return {
                "summary": summary
            }
            
        except Exception as e:
            from .spliter import SummaryError

            error_msg = extract_error_message(str(e))
            logger.error(f"内容分析失败: {error_msg}")

            # 根据错误类型给出针对性建议
            suggestions = get_error_suggestions(str(e), self.config.summary_model)

            raise SummaryError(error_msg, suggestions)
