---
title: Positional Encoding
related:
  - Transformer
  - Self-Attention
---
Self-attention is permutation-invariant: without position signals, the model cannot tell word order. **Positional encoding** injects order into token representations.

Common approaches:
- **Absolute sinusoids** — original Transformer
- **Learned absolute embeddings** — early GPT/BERT style
- **Relative positions** — T5, DeBERTa
- **RoPE (rotary)** — widely used in modern decoder LLMs

Good positional schemes improve length generalization and long-context behavior.
