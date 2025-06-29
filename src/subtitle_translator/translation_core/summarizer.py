from typing import Dict, Optional
from pathlib import Path
from openai import OpenAI
from .prompts import SUMMARIZER_PROMPT
from .config import SubtitleConfig
from .utils.json_repair import parse_llm_response
from .utils.logger import setup_logger

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
            # 使用 pathlib 处理文件名
            path = Path(input_file)
            # 获取不带扩展名的文件名
            readable_filename = path.stem.replace('_', ' ').replace('-', ' ')

            logger.info(f"可读性文件名: {readable_filename}")            
            # 更新提示词，强调文件名的权威性
            message = [
                {"role": "system", "content": (
                    "You are a precise subtitle summarizer. "
                    "When processing proper nouns and product names:"
                    "1. Use the filename as reference for product names"
                    "2. Only correct terms that appear to be ASR errors based on:"
                    "   - Similar pronunciation"
                    "   - Context indicating they refer to the same thing"
                    "3. Do not modify other technical terms or module names that are clearly different"
                    f"{SUMMARIZER_PROMPT}"
                )},
                {"role": "user", "content": f"Filename: {readable_filename}\n\nContent:\n{subtitle_content}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=message,
                temperature=0.7,
                timeout=80
            )
            
            summary = response.choices[0].message.content
            return {
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"总结字幕失败: {e}")
            return {
                "summary": ""
            }
