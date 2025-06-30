import os
import openai
import ast


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
    try:
        # 创建OpenAI客户端并发送请求到OpenAI API
        response = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=15).chat.completions.create(
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
                except:
                    # 如果ast.literal_eval失败，尝试JSON解析
                    try:
                        import json
                        error_dict = json.loads(error_json)
                        if "error" in error_dict and "message" in error_dict["error"]:
                            return False, error_dict["error"]["message"]
                    except:
                        pass
        except:
            pass
        return False, error_str


if __name__ == "__main__":
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = "google/gemini-2.0-flash-lite"
    success, msg = test_openai(base_url, api_key, model)
    print(f"Success: {success}, Message: {msg}")
