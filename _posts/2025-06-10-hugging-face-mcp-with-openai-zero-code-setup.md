---
layout: post
title: Hugging Face MCP with OpenAI - Zero Code Setup
date: 2025-06-10 20:25 +0800
comments: true
tags:
  - HuggingFace
  - MCP
  - OpenAI
---

Hugging Face just announced its experimental MCP server.  It allows AI agents to access Hugging Face's vast repository of models, datasets, and Spaces. It enables tasks like model information retrieval, dataset exploration, and NLP processing, making it a powerful tool for AI-driven applications.

In this blog, I'll show you show to do a zero-code setup from OpenAI ChatGPT to Hugging Face MCP Server.

### Zero-Code MCP Server Setup

![hf-mcp-server-token](/assets/images/uploads/hf-mcp/hf-mcp-openai-connect.png)

Head over to Hugging Face website and sign up for an account.  Then go to Settings > Access Tokens and create a new token.  Store this token securely on your localhost as we will need it for the next step.

Go to OpenAI platform -> Play Ground -> Prompts -> Tools -> MCP Server to add a new MCP server.

![hf-mcp-openai-mcp](/assets/images/uploads/hf-mcp/hf-mcp-openai-mcp.png)

Next fill in the necessary information:
- URL for HF Mcp server, which is: https://mcp.huggingface.co/mcp
- Your person access token saved from the previous step.

![hf-mcp-openai-connect](/assets/images/uploads/hf-mcp/hf-mcp-openai-connect.png)

Then select the access you want to grant.  I would recommend to grant all access for now.

Now let's take our new set up for a spin.  From the right-side of the OpenAI prompt, ask a question related to model, dataset etc. from Hugging Face.  For example, you can ask:

"NBA Dataset"

![hf-mcp-openai-prompt](/assets/images/uploads/hf-mcp/hf-mcp-openai-prompt.png)


Observe that when answering the prompt, OpenAI has included hf_mcp (tool) under assistant.  There is indeed one dataset for NBA, which is rare.

The real world use case is limitless due to the large variations of resources on Hugging Face.  One can use this setup to:
- Access Models: Query details about over 1 million models, including architectures and parameters.
- Explore Datasets: Retrieve metadata or content from thousands of datasets, such as NBA statistics or sentiment analysis corpora.
- Interact with Spaces: Run or query Hugging Face Spaces for demos like image generation or audio processing.
- Custom Integration: Connect with locally fine-tuned models or custom APIs for tailored workflows.

### Programmatic MCP Server Setup

One can certainly set this up programmatically with ChatGPT or with their own applications.

Frist, we will need to install required libraries:
`pip install mcp-client openai`

#### Set Up HF MCP Server Access:

Follow the official setup guide for hf-mcp-server in your code editor (e.g., VSCode, Cursor).
Configure your Hugging Face token:

```python
import os
os.environ["HF_ACCESS_TOKEN"] = "your_hf_token"
```

#### Initialize MCP Client:

Use the mcp-client library to connect to the HF MCP Server:

```python
from mcp_client import MCPClient
mcp = MCPClient(base_url="https://mcp.huggingface.co")
```

#### Integrate with ChatGPT:

Use the OpenAI API to send queries to ChatGPT, incorporating responses from the HF MCP Server:

```python
from openai import OpenAI
client = OpenAI(api_key="your_openai_api_key")

def query_hf_and_chatgpt(query):
    # Query HF MCP Server for relevant data
    hf_response = mcp.query(query)
    # Pass HF response to ChatGPT
    chatgpt_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Based on this Hugging Face data: {hf_response}, answer: {query}"}
        ]
    )
    return chatgpt_response.choices[0].message.content
```

#### Test the Integration:

Example query for model information:

```python
query = "Get details about the BERT model on Hugging Face."
response = query_hf_and_chatgpt(query)
print(response)
```

Output might include BERT’s architecture, parameters, and use cases, processed by ChatGPT for clarity.


**The MCP Server allows you to fetch**:

- Model Metadata: Details like model size, training data, or performance metrics.
- Dataset Information: Metadata, samples, or full datasets for tasks like NLP, vision, or audio.
- Spaces: Interact with demo apps for tasks like image editing or speech recognition.

**Example: Fetching NBA Dataset**

Query the Dataset:

```python
query = "Find the NBA dataset on Hugging Face."
hf_response = mcp.query(query)
print(hf_response)
```

This might return metadata for a dataset like nba_player_stats (hypothetical), including player performance metrics or game statistics.


Process with ChatGPT:
```python
chatgpt_response = query_hf_and_chatgpt("Summarize the NBA dataset: " + str(hf_response))
print(chatgpt_response)
```

ChatGPT could summarize: "The NBA dataset contains player stats like points, rebounds, and assists for the 2023-2024 season, useful for sports analytics."



**Other Interesting Datasets**

Yelp Reviews: For sentiment analysis or text classification.
```python
query = "Load the Yelp Review dataset."
hf_response = mcp.query(query)
```

Wikitext-2: Wikipedia articles for training language models.

```python
query = "Get details about the Wikitext-2 dataset."
hf_response = mcp.query(query)
```

OpenAssistant Conversations (OASST1): Conversational data for chatbot training.

```python
query = "Explore the OpenAssistant Conversations dataset."
hf_response = mcp.query(query)
```

Enjoy this wonderful new feature from HuggingFace!