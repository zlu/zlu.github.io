---
title: ReLU
synonyms:
  - Rectified Linear Unit
related:
  - Activation Function
  - MLP
  - Feed-Forward Network
---
**ReLU** (rectified linear unit) is $f(x) = \max(0, x)$. It is cheap, induces sparsity, and historically unlocked deeper nets versus saturating sigmoids/tanhs.

### Variants
- **Leaky / PReLU** — small slope for $x < 0$ to reduce “dead” units
- **GELU** — smooth Gaussian-error gating; common in BERT-style and many LLMs
- **SwiGLU / SiLU** — gated GLU-style activations in modern transformer FFNs

### Practice
Plain ReLU is still fine for many vision MLPs and simple heads. Transformer [feed-forward](/ai-glossary/feed-forward-network/) blocks in current LLMs usually prefer GELU or SwiGLU over vanilla ReLU. See [activation function](/ai-glossary/activation-function/) for the broader family.
