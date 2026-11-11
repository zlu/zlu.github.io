---
title: Positional Encoding
related:
  - Transformer
  - Self-Attention
  - Embedding
  - Context Window
  - Decoder-Only
---
[Self-attention](/ai-glossary/self-attention/) is permutation-invariant: without position signals, the model cannot tell word order. **Positional encoding** injects order into token representations—usually by adding or rotating vectors derived from each index.

### Common approaches
- **Absolute sinusoids** — original Transformer; fixed $\sin$/$\cos$ functions of position
- **Learned absolute embeddings** — early GPT/BERT style; a table of position vectors
- **Relative positions** — T5, DeBERTa; bias or embeddings on pairwise distances
- **RoPE (rotary)** — widely used in modern decoder LLMs; rotates Q/K by position

Good positional schemes improve length generalization and long-context behavior inside the [context window](/ai-glossary/context-window/). Extrapolating far beyond training lengths still needs careful design (NTK-aware scaling, ALiBi, interpolation). Position is part of the input representation alongside the token [embedding](/ai-glossary/embedding/)—not a separate “module” you can ignore at serve time.
