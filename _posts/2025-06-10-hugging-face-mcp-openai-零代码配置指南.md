---
layout: post
title: "Hugging Face MCP 与 OpenAI 集成 - 零代码配置指南"
date: 2025-06-10 20:25 +0800
comments: true
tags:
    - Hugging Face
    - MCP
    - OpenAI
    - AI
    - 零代码
    - 集成
---

Hugging Face 最近发布了其实验性的 MCP 服务器。它允许 AI 智能体访问 Hugging Face 庞大的模型库、数据集和 Spaces。它支持模型信息检索、数据集探索和自然语言处理等任务，成为 AI 驱动应用程序的强大工具。

在这篇博客中，我将向您展示如何从 OpenAI ChatGPT 到 Hugging Face MCP 服务器进行零代码配置。

### 零代码 MCP 服务器配置

![hf-mcp-server-token](/assets/images/uploads/hf-mcp/hf-mcp-openai-connect.png)

访问 Hugging Face 网站并注册一个账户。然后进入 设置 > 访问令牌 创建一个新令牌。请将此令牌安全地保存在本地，因为我们下一步会需要它。

前往 OpenAI 平台 -> Play Ground -> Prompts -> Tools -> MCP Server 添加新的 MCP 服务器。

![hf-mcp-openai-mcp](/assets/images/uploads/hf-mcp/hf-mcp-openai-mcp.png)

填写必要信息：
- HF MCP 服务器的 URL：https://mcp.huggingface.co/mcp
- 上一步保存的个人访问令牌

![hf-mcp-openai-connect](/assets/images/uploads/hf-mcp/hf-mcp-openai-connect.png)

然后选择您想要授予的访问权限。我建议现在先授予所有访问权限。

现在让我们来测试一下新配置。在 OpenAI 提示框的右侧，询问与 Hugging Face 上的模型、数据集等相关的问题。例如，您可以问：

"NBA 数据集"

![hf-mcp-openai-prompt](/assets/images/uploads/hf-mcp/hf-mcp-openai-prompt.png)

在回答提示时，您会注意到 OpenAI 在助手下包含了 hf_mcp（工具）。确实有一个 NBA 数据集，这很罕见。

由于 Hugging Face 上海量的资源，实际应用场景是无限的。您可以使用此设置来：
- 访问模型：查询超过 100 万个模型的详细信息，包括架构和参数。
- 探索数据集：从数千个数据集中检索元数据或内容，例如 NBA 统计数据或情感分析语料库。
- 与 Spaces 交互：运行或查询 Hugging Face Spaces 以获取演示，如图像生成或音频处理。
- 自定义集成：与本地微调模型或自定义 API 连接，实现定制化工作流。

### 编程方式配置 MCP 服务器

当然，您也可以通过编程方式使用 ChatGPT 或自己的应用程序来设置。

首先，我们需要安装所需的库：
`pip install mcp-client openai`

#### 设置 HF MCP 服务器访问：

在代码编辑器（如 VSCode、Cursor）中按照官方指南设置 hf-mcp-server。
配置您的 Hugging Face 令牌：

```python
import os
os.environ["HF_ACCESS_TOKEN"] = "your_hf_token"
```

#### 初始化 MCP 客户端：

使用 mcp-client 库连接到 HF MCP 服务器：

```python
from mcp_client import MCPClient
mcp = MCPClient(base_url="https://mcp.huggingface.co")
```

#### 与 ChatGPT 集成：

使用 OpenAI API 向 ChatGPT 发送查询，并整合来自 HF MCP 服务器的响应：

```python
from openai import OpenAI
client = OpenAI(api_key="your_openai_api_key")

def query_hf_and_chatgpt(query):
    # 查询 HF MCP 服务器获取相关数据
    hf_response = mcp.query(query)
    # 将 HF 响应传递给 ChatGPT
    chatgpt_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Based on this Hugging Face data: {hf_response}, answer: {query}"}
        ]
    )
    return chatgpt_response.choices[0].message.content
```

#### 测试集成：

查询模型信息的示例：

```python
query = "获取 Hugging Face 上 BERT 模型的详细信息。"
response = query_hf_and_chatgpt(query)
print(response)
```

输出可能包括 BERT 的架构、参数和用例，这些信息都经过 ChatGPT 处理以提高可读性。

**MCP 服务器允许您获取**：

- 模型元数据：如模型大小、训练数据或性能指标等详细信息。
- 数据集信息：用于 NLP、视觉或音频等任务的元数据、样本或完整数据集。
- Spaces：与演示应用交互，如图像编辑或语音识别。

**示例：获取 NBA 数据集**

查询数据集：

```python
query = "在 Hugging Face 上查找 NBA 数据集。"
hf_response = mcp.query(query)
print(hf_response)
```

这可能会返回类似 nba_player_stats（假设）的数据集元数据，包括球员表现指标或比赛统计数据。

使用 ChatGPT 处理：
```python
chatgpt_response = query_hf_and_chatgpt("总结 NBA 数据集：" + str(hf_response))
print(chatgpt_response)
```

ChatGPT 可能会总结为："NBA 数据集包含 2023-2024 赛季的球员统计数据，如得分、篮板和助攻，适用于体育分析。"


**其他有趣的数据集**

Yelp 评论：用于情感分析或文本分类。
```python
query = "加载 Yelp 评论数据集。"
hf_response = mcp.query(query)
```

Wikitext-2：用于训练语言模型的维基百科文章。

```python
query = "获取 Wikitext-2 数据集的详细信息。"
hf_response = mcp.query(query)
```

OpenAssistant 对话 (OASST1)：用于聊天机器人训练的对话数据。

```python
query = "探索 OpenAssistant 对话数据集。"
hf_response = mcp.query(query)
```

尽情享受 HuggingFace 这个精彩的新功能吧！
