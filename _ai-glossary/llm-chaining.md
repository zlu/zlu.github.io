---
title: LLM Chaining
description: "LLM chaining connects multiple model calls so each step’s output feeds the next. Used to split research, drafting, and formatting into controlled stages."
keywords: "llm chaining, LLM chain, prompt chain, sequential LLM, agent pipeline"
related:
  - Revise&Reflect
  - RAG
---
**LLM chaining** connects multiple language-model calls so the output of one step becomes the input to the next. Each call has a narrow job.

#### Example: blog draft from a topic

1. Summarize key points about the topic → bullets  
2. Expand each bullet into a paragraph → body  
3. Write a title and meta description from the body → SERP fields  

#### Why chain

- Split hard tasks into smaller steps  
- Isolate reasoning from formatting  
- Reuse steps across products  

#### Variants

- **Sequential:** fixed step order  
- **Conditional:** branch on a runtime decision  
- **Parallel:** independent calls, then merge  

Related product pattern: [RAG](/ai-glossary/rag/) retrieves context before a generate step; chaining can wrap retrieval, drafting, and citation checks as separate calls.
