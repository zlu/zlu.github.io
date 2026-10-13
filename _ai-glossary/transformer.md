---
title: Transformer
synonyms:
  - Transformer Architecture
related:
  - Attention
  - Self-Attention
  - Multi-Head Attention
  - Positional Encoding
---
The **Transformer** is a neural network architecture introduced in *Attention Is All You Need* (Vaswani et al., 2017). It replaces recurrence with **self-attention**, enabling parallel training over sequences and becoming the backbone of modern LLMs (GPT, BERT, LLaMA, Qwen, and others).

### Building blocks
- **Self-attention** — every token can attend to every other token in the context
- **Multi-head attention** — several attention subspaces in parallel
- **Feed-forward layers** — position-wise MLPs after attention
- **Residual connections + layer norm** — stabilize deep stacks
- **Positional encoding** — inject order information (absolute, relative, or RoPE)

### Why it matters
Transformers scale well with data and compute, support long-context modeling (with caveats), and unify encoder-only, decoder-only, and encoder–decoder designs used across NLP and multimodal models.
