---
title: Mixture of Experts
synonyms:
  - MoE
related:
  - Feed-Forward Network
  - Transformer
  - Inference
  - Parameter
---
**Mixture of Experts (MoE)** replaces (or augments) a dense [feed-forward network](/ai-glossary/feed-forward-network/) with many expert FFNs plus a router that sends each token to a small subset (often top-1 or top-2). Total [parameter](/ai-glossary/parameter/) count grows a lot; FLOPs per token grow much less.

### Tradeoffs
- Higher capacity at fixed active compute
- Needs load balancing so experts are not idle or overloaded
- Serving is harder: expert placement, all-to-all communication, capacity factors

### Where you see it
Sparse MoE [transformers](/ai-glossary/transformer/) (e.g., Mixtral-style) aim for strong quality per FLOP. At [inference](/ai-glossary/inference/), latency and memory depend on how many experts are loaded and how routing batches. Dense models remain simpler to train and deploy when budget allows.
