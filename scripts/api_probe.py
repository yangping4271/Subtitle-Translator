import os
import ast

from subtitle_translator.translation_core.config import SubtitleConfig
from subtitle_translator.translation_core.llm_client import LLMClient


def test_openai(base_url, api_key, model):
    """
    这是一个测试OpenAI API的函数。
    它使用指定的API设置与OpenAI的GPT模型进行对话。

    参数:
    user_message (str): 用户输入的消息

    返回:
    bool: 是否成功
    str: 错误信息或者AI助手的回复
    """
    client = None
    try:
        # 复用项目统一客户端，确保探测请求也执行禁用思考策略。
        config = SubtitleConfig(
            openai_base_url=base_url,
            openai_api_key=api_key,
            disable_thinking=True,
        )
        client = LLMClient(config)
        response = client.create_chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=100,
            timeout=10
        )
        # 返回AI的回复
        return True, str(response.choices[0].message.content)
    except Exception as e:
        error_str = str(e)
        # 提取核心错误信息
        try:
            if " - " in error_str:
                error_json = error_str.split(" - ", 1)[1]
                try:
                    # 尝试使用ast.literal_eval解析Python字典
                    error_dict = ast.literal_eval(error_json)
                    if "error" in error_dict and "message" in error_dict["error"]:
                        return False, error_dict["error"]["message"]
                except Exception:
                    # 如果ast.literal_eval失败，尝试JSON解析
                    try:
                        import json
                        error_dict = json.loads(error_json)
                        if "error" in error_dict and "message" in error_dict["error"]:
                            return False, error_dict["error"]["message"]
                    except Exception:
                        pass
        except Exception:
            pass
        return False, error_str
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL") or os.getenv("TRANSLATION_MODEL")
    success, msg = test_openai(base_url, api_key, model)
    print(f"Success: {success}, Message: {msg}")
