---
layout: post
title: RAG Based Knowledge Base
date: 2025-06-20
comments: true
tags:
  - RAG
  - Knowledgebase
---



## RAG Architecture Overview

Retrieval-Augmented Generation (RAG) combines the power of large language models with a knowledge base to provide accurate, context-aware responses. Here's the architecture breakdown:

1. Document Processing Layer
   - Supports multiple document types (PDF, TXT, EPUB)
   - Documents are stored in a configurable Knowledge Base directory
   - Uses specialized loaders for each file type (PyPDF2, TextLoader, EbookLib)

2. Text Processing Layer
   - Splits documents into manageable chunks using RecursiveCharacterTextSplitter
   - Default: 1000 characters per chunk with 200 character overlap
   - Maintains source metadata for each chunk

3. Embedding Layer
   - Uses OllamaEmbeddings with "nomic-embed-text" model
   - Converts text chunks into vector representations
   - Optimized for semantic similarity matching

4. Vector Storage Layer
   - Supports multiple vector databases:
     - Chroma (default): Persistent storage in "chroma_db"
     - FAISS: Index stored in "faiss_index"
   - Stores both embeddings and metadata

5. Retrieval & Generation Layer
   - Similarity search finds relevant chunks (top-4 by default)
   - Local LLM (Deepseek via Ollama) processes query with context
   - Returns answer with source documentation

6. Real-time Update System
   - FileWatcher monitors Knowledge Base directory
   - Automatically rebuilds index on document changes
   - Ensures knowledge base stays current

## RAG Based Knowledge:

Agent is a smart assistant that knows what tool to use, when to use it, and how to combine outputs to answer a complex question.

Using Agent in a RAG Pipeline

With only RAG, the pipeline does the following:

User: "Summerize company's HR policy on sick leave."

- Embed the user query.
- Retrieve top-3 chunks from a vector store.
- Send retrieved chunks with query to LLM.
- Return answer.


With Agent assisted RAG, the pipeline does the following:

User: "Compare our company's HR policy on sick leave with competitors."

- Agent sees: this needs **company KB** and **external lookup**.
- Agent calls Tool A, such as searching local vector DB for HR policy.
- Agent calls Took B, such as a web search or fetch from external JSON.
- Agent merges results and uses LLM to analyze and compare.
- Agent returns answer, possibly with citations.

With agent, we can solve more sophisticated questions over multiple steops which is not possible in a pure RAG system.

Here's the flow:

![agent-rag-flow](/assets/images/uploads/rag-agent.png)






![kb-rag-langchain](/assets/images/uploads/kb-rag-langchain.png)

## Implementation Details

Let's look at how to implement a RAG system using LangChain and local LLMs.

### What is LangChain?

LangChain is a framework for developing applications powered by language models. It provides:
1. Components for common operations (loading documents, splitting text, managing prompts)
2. Integration with various LLMs, embedding models, and vector stores
3. Tools for building chains and agents

### Basic RAG Implementation with Local LLM

Instead of using OpenAI's API (which requires an API key and is not free), we can use locally installed models via Ollama. Here's how to implement a basic RAG system using the Deepseek model:

1. First, install Deepseek in Ollama:
```bash
ollama pull deepseek-coder:1.3b
```

2. Basic LangChain workflow for RAG:

```python
# 1. Load documents
from langchain_community.document_loaders import TextLoader
loader = TextLoader("file.txt")
docs = loader.load()

# 2. Split documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Create embeddings and store in vector store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Set up retrieval chain with local LLM
from langchain_community.llms import Ollama

llm = Ollama(model="deepseek-coder:1.3b")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5. Ask questions
question = "What is this document about?"
answer = qa_chain.run(question)
```

This implementation:
1. Takes the user query
2. Retrieves relevant documents using the retriever (which gets the most similar chunks)
3. "Stuffs" these documents into the prompt along with the query
4. Sends this context-enriched prompt to the local LLM via Ollama

The `RetrievalQA` chain from LangChain handles combining the context with the query and managing the interaction with the LLM. While this is a basic implementation, it provides a foundation that can be enhanced with:
- Custom prompt templates
- Better context preprocessing and selection
- Advanced retrieval strategies
- Source validation and citation requirements

## Current Limitations

1. Context Handling
   - Basic "stuff" method simply concatenates retrieved chunks
   - No smart context prioritization or filtering
   - Limited by context window size of the LLM
   - May miss important information if not in top-k retrieved chunks

2. Retrieval Strategy
   - Simple similarity search might not capture complex relationships
   - No hybrid search combining keyword and semantic matching
   - Lacks support for structured queries or filters
   - No handling of temporal aspects or document freshness

3. Prompt Engineering
   - Uses default LangChain prompts
   - No custom instruction for domain-specific tasks
   - Limited control over response format and style
   - No system-level prompt optimization

4. Quality Control
   - No fact-checking or hallucination detection
   - Missing source attribution in responses
   - No confidence scoring for retrieved context
   - Lacks validation of response against source material

5. Performance
   - Sequential processing of documents during indexing
   - Full index rebuild on any document change
   - No caching mechanism for frequent queries
   - Resource-intensive for large document collections
these are some junkieee

k
6. User Experience
   - Basic chat interface without advanced features
   - No explanation of reasoning or source citations
   - Limited feedback mechanisms
   - No conversation memory or context preservation

These limitations provide opportunities for future improvements, such as implementing hybrid search, adding conversation memory, improving context handling, and enhancing the quality control mechanisms.
