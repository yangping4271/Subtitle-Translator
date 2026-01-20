# 开发规范

## Prompt 管理

### 占位符风格

统一使用 `{placeholder}` 格式，通过 `.format()` 替换：

```python
prompt = "翻译成 {target_language}：{text}"
result = prompt.format(target_language="中文", text="Hello")
```

### JSON 示例转义

Prompt 中的 JSON 示例需要双花括号：

```python
prompt = """
返回 JSON 格式：
{{
  "translation": "翻译结果"
}}
"""
```

### 术语表外置

不要硬编码术语，使用 `terminology.py` 管理。

### 逻辑集中

所有 prompt 相关规则写在 `prompts.py`，不要分散。

## 自定义术语表

编辑 `src/subtitle_translator/translation_core/terminology.py`：

```python
DEFAULT_TERMINOLOGY = {
    "简体中文": {
        "AGI": "通用人工智能 (AGI)",
        "LLM": "大语言模型 (Large Language Model)",
    },
    "繁体中文": {
        "AGI": "通用人工智慧 (AGI)",
    },
    "日文": {
        "AGI": "汎用人工知能 (AGI)",
    }
}
```

## 其他规范

- 开发模式（`-e`）代码修改立即生效
