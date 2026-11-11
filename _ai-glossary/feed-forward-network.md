---
title: Feed-Forward Network
synonyms:
  - FFN
  - MLP Block
related:
  - MLP
  - Transformer
  - Activation Function
  - Mixture of Experts
---
In [Transformers](/ai-glossary/transformer/), the **feed-forward network** (position-wise [MLP](/ai-glossary/mlp/)) follows attention in each block. It expands each token’s hidden state to a wider intermediate size, applies a nonlinearity ([ReLU](/ai-glossary/relu/), GeLU, SwiGLU, …), then projects back to model dimension.

### Why it matters
Much of an LLM’s parameter count lives in FFNs—often more than attention projections. Capacity for factual and syntactic “storage” is frequently attributed to these layers. [MoE](/ai-glossary/mixture-of-experts/) variants sparsely activate expert FFNs per token to scale parameters without full dense compute.

### Design knobs
- Intermediate multiplier (e.g., $4\times$ or gated variants)
- Activation choice and whether weights are shared across positions (they are—same FFN per position)
- [LoRA](/ai-glossary/lora/) / adapters often target FFN projections for domain adaptation

FFNs are applied independently per position; all cross-token mixing happens in attention.
