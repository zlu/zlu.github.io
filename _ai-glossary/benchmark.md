---
title: Benchmark
related:
  - Evaluation
  - Massive Multitask Language Understanding
---
A **benchmark** is a standardized dataset plus metric used to compare models—e.g. [MMLU](/ai-glossary/massive-multitask-language-understanding/), HumanEval, GSM8K, BEIR for retrieval. Shared suites make progress trackable across papers and releases.

### Why it matters
- Enables apples-to-apples comparison when protocols match
- Easy to overfit, leak into pretraining, or misread as product readiness
- Leaderboard gains may not transfer to your domain distribution

### How it shows up
Report the exact split, prompt format, and few-shot setting—small protocol drifts move scores a lot. Pair public benchmarks with private, task-specific [evaluation](/ai-glossary/evaluation/) for shipping decisions. Treat saturation and contamination as first-class risks: if the model may have seen the test items, the number is marketing, not measurement.
