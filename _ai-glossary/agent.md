---
title: Agent
synonyms:
  - AI Agent
  - LLM Agent
related:
  - Tool Calling
  - RAG
  - LLM Chaining
  - Prompt Injection
---
An **LLM agent** uses a language model as planner/controller: it proposes actions, calls tools (search, code, APIs, browsers), observes results, and iterates toward a goal instead of answering in one shot. [Tool calling](/ai-glossary/tool-calling/), [RAG](/ai-glossary/rag/), and multi-step [chaining](/ai-glossary/llm-chaining/) are the usual building blocks.

### Why it matters
- Unlocks workflows that need external state or computation
- Amplifies failure modes: loops, tool misuse, cost blowups, [prompt injection](/ai-glossary/prompt-injection/)
- Quality depends as much on tool design and stop conditions as on the model

### How it shows up
Typical loop: plan → call tool → append observation → replan until done or max steps. Keep tools narrowly scoped, require confirmation for irreversible actions, cap iterations, and log every step. Evaluate agents on task success and safety traces, not only single-turn chat benchmarks.
