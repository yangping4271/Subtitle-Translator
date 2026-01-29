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

术语表使用外部文本文件管理，不要硬编码在代码中。

### 逻辑集中

所有 prompt 相关规则写在 `prompts.py`，不要分散。

## 自定义术语表

### 全局术语表

编辑用户配置目录的术语表：

```bash
# 创建配置目录
mkdir -p ~/.config/subtitle-translator

# 编辑全局术语表
vim ~/.config/subtitle-translator/terminology.txt
```

格式：

```
# 全局术语表
[简体中文]
AGI = 通用人工智能 (AGI)
LLM = 大语言模型 (Large Language Model)

[繁体中文]
AGI = 通用人工智慧 (AGI)

[日文]
AGI = 汎用人工知能 (AGI)
```

### 局部术语表

在字幕文件同目录创建 `terminology.txt`，局部术语会与全局术语合并：

```
# 局部术语表
[简体中文]
# 覆盖全局术语
AGI = 人工通用智能 (AGI)
# 新增项目特定术语
project-term = 项目术语
```

**格式说明**：
- 支持 `#` 开头的注释行
- 使用 `[语言]` 标记语言段
- 使用 `术语 = 翻译` 格式
- 局部术语会覆盖全局术语

## 其他规范

- 开发模式（`-e`）代码修改立即生效
