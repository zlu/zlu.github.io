---
title: Massive Multitask Language Understanding
---
It is a benchmark in machine learning and natural language processing (NLP) used to evaluate the general language understanding and reasoning ability of large language models (LLMs).

It was introduced in 2021 by Hendrycks et al. to test a model’s performance across a broad set of knowledge-rich and reasoning-heavy tasks.  It uses accuracy (percentage of correct answers) to evaluate multiple-choice questions.

MMLU is designed to go beyond simple pattern recognition and test if a model can handle:
- Factual knowledge
- Reasoning ability
- Multitask generalization across topics

It’s widely used to benchmark LLMs like GPT-4, Claude, PaLM, LLaMA, etc.

#### What’s in the Benchmark?
- 57 diverse tasks
- Divided into 4 main categories:
  1. Humanities (e.g., history, law)
  2. STEM (e.g., physics, computer science)
  3. Social Sciences (e.g., economics, psychology)
  4. Other (e.g., professional law, medical exams)

Each task has a training/test/dev split but the MMLU benchmark only evaluates on test data using zero-shot or few-shot prompting.

#### Example Question (from MMLU - Physics):
```
What is the unit of electric resistance?
A) Volt
B) Ampere
C) Ohm
D) Watt

Correct Answer: C) Ohm
```
