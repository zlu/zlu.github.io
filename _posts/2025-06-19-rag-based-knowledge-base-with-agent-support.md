---
layout: post
title: Privacy-Aware RAG Based Knowledge Base
date: 2025-06-20
comments: true
tags:
  - RAG
  - Knowledgebase
---

In this blog, we will discuss how to build a local RAG-based knowledge base that protects both privacy and economics.  Local means that the documents to be analyzed will not get processed in any cloud-based LLMs.  It alsoIn this blog, we will discuss how to create a local RAG-based knowledge base that prioritizes both privacy and cost-effectiveness. By "local," we mean that the documents being analyzed will not be processed through any cloud-based language models. This approach also eliminates the need for API calls to services like OpenAI, resulting in cost savings. 

We will start with a basic RAG implementation using Langchain and an open-source embedding model. Next, we will incorporate agent support to enhance the intelligence of the RAG. Finally, we will make improvements to optimize the RAG's performance. means that no API calls will be made against OpenAI and similar services, thus saving money.  We will start with a basic RAG based on Langchain and Opensource embedding model.  Then, we will add
the support of the Agent to make the RAG smarter.  Lastly, we will enhance the RAG for improved performance.


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

### Architecture Diagrams

The following diagrams illustrate the complete RAG architecture and agent-assisted flow:

#### Six-Layer RAG Architecture

<div class="mermaid">
graph TB
    subgraph "1. Document Processing Layer"
        A1[Knowledge Base Directory]
        A2[PDF Loader]
        A3[TXT Loader]
        A4[EPUB Loader]
        A1 --> A2
        A1 --> A3
        A1 --> A4
    end

    subgraph "2. Text Processing Layer"
        B1[RecursiveCharacterTextSplitter]
        B2[1000 chars/chunk]
        B3[200 char overlap]
        B4[Metadata Preservation]
        B1 --> B2
        B1 --> B3
        B1 --> B4
    end

    subgraph "3. Embedding Layer"
        C1[OllamaEmbeddings]
        C2[nomic-embed-text model]
        C3[Vector Representations]
        C1 --> C2
        C2 --> C3
    end

    subgraph "4. Vector Storage Layer"
        D1[Vector Databases]
        D2[Chroma DB]
        D3[FAISS Index]
        D4[Metadata Storage]
        D1 --> D2
        D1 --> D3
        D2 --> D4
        D3 --> D4
    end

    subgraph "5. Retrieval & Generation Layer"
        E1[Similarity Search]
        E2[Top-4 Results]
        E3[Local LLM/Deepseek]
        E4[Context Processing]
        E5[Answer Generation]
        E1 --> E2
        E2 --> E3
        E3 --> E4
        E4 --> E5
    end

    subgraph "6. Real-time Update System"
        F1[FileWatcher]
        F2[Auto Index Rebuild]
        F3[Knowledge Base Updates]
        F1 --> F2
        F2 --> F3
    end

    A4 --> B1
    B4 --> C1
    C3 --> D1
    D4 --> E1
    F3 --> A1
</div>

#### Agent-Assisted RAG Flow

<div class="mermaid">
sequenceDiagram
    participant U as User
    participant A as Agent
    participant KB as Knowledge Base
    participant T as Tools
    participant LLM as Local LLM
    participant V as Vector Store

    U->>A: Complex Query
    A->>A: Analyze Query Requirements
    
    par Tool Execution
        A->>KB: Search Local KB
        KB->>V: Vector Search
        V-->>KB: Relevant Chunks
        KB-->>A: Local Context
        and External Search
        A->>T: External Tool Calls
        T-->>A: External Data
    end

    A->>LLM: Merge & Process Results
    LLM-->>A: Analyzed Response
    A->>U: Final Answer with Citations

    Note over A,LLM: Agent orchestrates multiple tools<br/>and knowledge sources
</div>

Now that we understand the architecture and flow of our RAG system, let's explore the implementation details using LangChain and local LLMs.

## Implementation Details

Let's look at how to implement a RAG system using LangChain and local LLMs.

### What is LangChain?

LangChain is a framework for developing applications powered by language models. It provides:
1. Components for common operations (loading documents, splitting text, managing prompts)
2. Integration with various LLMs, embedding models, and vector stores
3. Tools for building chains and agents

With this foundation, we can now implement our RAG system, starting with the basic components and then adding agent capabilities.

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

## Advanced Agent-RAG Implementation

Building on the agent-enhanced approach described above, let's implement a comprehensive system that leverages these capabilities:

### Agent Tools and Capabilities

The power of Agent-RAG comes from its ability to use different tools for different tasks. Here's how to implement common tools:

```python
from langchain.tools import BaseTool
from typing import Optional

class LocalKnowledgeBaseTool(BaseTool):
    name = "local_kb_search"
    description = "Search local knowledge base for relevant information"

    def _run(self, query: str) -> str:
        results = vectorstore.similarity_search(query, k=4)
        return self._format_results(results)

    def _format_results(self, results) -> str:
        return "\n".join([doc.page_content for doc in results])

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for supplementary information"

    def _run(self, query: str) -> str:
        # Implement web search logic
        return web_results

# Tool Orchestration
class ToolOrchestrator:
    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}
        self.fallback_tool = None

    def execute_with_fallback(self, tool_name: str, query: str) -> str:
        try:
            return self.tools[tool_name]._run(query)
        except Exception as e:
            if self.fallback_tool:
                return self.fallback_tool._run(query)
            raise e
```

### Performance Optimization

Implement these optimization strategies to improve RAG performance:

```python
# 1. Parallel Processing
from concurrent.futures import ThreadPoolExecutor

def parallel_document_processing(documents: List[Document]):
    with ThreadPoolExecutor() as executor:
        chunks = executor.map(text_splitter.split_text, documents)
    return list(chunks)

# 2. Caching System
from functools import lru_cache

class CachedVectorStore:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.cache = {}

    @lru_cache(maxsize=1000)
    def similarity_search(self, query: str, k: int = 4):
        return self.vectorstore.similarity_search(query, k=k)

# 3. Index Optimization
def optimize_index(vectorstore):
    if isinstance(vectorstore, FAISS):
        vectorstore.index.nprobe = 8  # Adjust based on index size
    return vectorstore

# 4. Query Refinement
def refine_query(query: str, context: Optional[str] = None) -> str:
    refined = llm(f"""Improve this search query while maintaining its original intent.
    Original query: {query}
    Context: {context or 'None'}
    Improved query:""")
    return refined.strip()
```

### Real-World Usage Patterns

Here are some practical patterns for common scenarios:

```python
class EnhancedRAGAgent:
    def __init__(self, tools, llm, vectorstore):
        self.tools = tools
        self.llm = llm
        self.vectorstore = vectorstore
        self.history = []

    def process_query(self, query: str) -> str:
        # 1. Query Analysis
        query_type = self.analyze_query_type(query)
        
        # 2. Tool Selection
        selected_tools = self.select_tools(query_type)
        
        # 3. Parallel Information Gathering
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda tool: tool._run(query),
                selected_tools
            ))
        
        # 4. Context Management
        context = self.merge_results(results)
        context = self.filter_relevant_context(context, query)
        
        # 5. Response Generation
        response = self.generate_response(query, context)
        
        # 6. Add Citations
        final_response = self.add_citations(response, results)
        
        self.history.append((query, final_response))
        return final_response

    def filter_relevant_context(self, context: str, query: str) -> str:
        relevance_scores = self.calculate_relevance(context, query)
        return self.select_top_relevant(context, relevance_scores)

    def add_citations(self, response: str, results: List[str]) -> str:
        sources = self.extract_sources(results)
        return f"{response}\n\nSources:\n" + "\n".join(sources)
```

Example usage in a real-world scenario:

```python
# Initialize the enhanced RAG system
agent = EnhancedRAGAgent(
    tools=[
        LocalKnowledgeBaseTool(),
        WebSearchTool(),
        ImageAnalysisTool(),
    ],
    llm=Ollama(model="deepseek-coder:1.3b"),
    vectorstore=CachedVectorStore(
        optimize_index(
            Chroma(embedding_function=OllamaEmbeddings())
        )
    )
)

# Process a complex query
response = agent.process_query(
    "Compare our company's sick leave policy with industry standards, "
    "focusing on tech companies in the Bay Area."
)
```

This implementation:
- Uses multiple tools in parallel
- Handles errors gracefully
- Provides source citations
- Optimizes performance with caching
- Manages context effectively
- Generates well-structured responses

## Future Challenges and Opportunities

1. Advanced Context Understanding
   - Handling complex logical relationships between documents
   - Understanding and preserving document hierarchies
   - Cross-document coreference resolution
   - Dynamic context window management for different LLMs

2. Quality and Trust
   - Real-time fact verification mechanisms
   - Confidence scoring for generated responses
   - Bias detection and mitigation
   - Privacy-preserving information retrieval

3. Multi-Modal Support
   - Handling of images, audio, and video content
   - Cross-modal reasoning and retrieval
   - Multi-modal answer generation
   - Structured data integration

4. Advanced Interaction
   - Interactive clarification dialogues
   - Multi-turn reasoning capabilities
   - Explanation of agent decision-making
   - User preference learning

These challenges represent exciting opportunities for future development in RAG systems. As the field evolves, addressing these areas will lead to more sophisticated and capable knowledge management systems that can better serve users' needs while maintaining privacy and trust.
