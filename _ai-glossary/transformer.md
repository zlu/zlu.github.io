---
title: Transformer
synonyms:
  - Transformer Architecture
related:
  - Attention
  - Self-Attention
  - Multi-Head Attention
  - Positional Encoding
  - Encoder–Decoder
  - Decoder-Only
  - Feed-Forward Network
  - Residual Connection
---
The **Transformer** is a neural network architecture introduced in *Attention Is All You Need* (Vaswani et al., 2017). It replaces recurrence with **[self-attention](/ai-glossary/self-attention/)**, enabling parallel training over sequences and becoming the backbone of modern LLMs (GPT, BERT, LLaMA, Qwen, and others).

### Building blocks
- **Self-attention** — every token can attend to every other token in the context
- **[Multi-head attention](/ai-glossary/multi-head-attention/)** — several attention subspaces in parallel
- **Feed-forward layers** — position-wise MLPs after attention
- **[Residual connections](/ai-glossary/residual-connection/) + layer norm** — stabilize deep stacks
- **[Positional encoding](/ai-glossary/positional-encoding/)** — inject order (absolute, relative, or RoPE)

### Layouts
- **Encoder-only** — bidirectional context (BERT-style)
- **Decoder-only** — causal LM stack used by most chat models
- **[Encoder–decoder](/ai-glossary/encoder-decoder/)** — classic seq2seq (T5, BART, MT)

Transformers scale well with data and compute, support long-context modeling (with quadratic attention caveats), and unify text and multimodal stacks behind the same attention + MLP block.
