# Terminology And Context

## 术语表

全局：

```text
~/.config/subtitle-translator/terminology.txt
```

局部：

```text
<字幕目录>/terminology.txt
```

默认优先维护全局术语表，只有目录特有规则才写局部。

## alias 规则

格式：

```text
[简体中文]
LangChain = LangChain | aliases: land chain, lang chain
```

原则：

- alias 记录真实出现过的 ASR 误识别
- 标准术语翻译不能只靠 alias 替代
- 外部 glossary 不能替代 alias

## 上下文文件

可用文件：

```text
context.txt
ctx.txt
```

只写翻译决策需要的信息，例如：

- 课程名
- 产品名
- 品牌名
- 讲者
- 专业领域
