---
title: Activation Function
related:
  - ReLU
  - Softmax
  - Neural Network
  - Feed-Forward Network
---
An **activation function** applies an elementwise nonlinearity after a linear transform so stacked layers can approximate non-linear maps. Without activations, a deep net collapses to one big linear model.

### Common choices
- **[ReLU](/ai-glossary/relu/)** / GELU / SiLU (Swish) — hidden layers; Transformers often use GELU or SwiGLU in the [FFN](/ai-glossary/feed-forward-network/)
- **tanh / sigmoid** — older nets and gates; saturating if misused
- **[Softmax](/ai-glossary/softmax/)** — output distributions over classes or vocabulary tokens

### Why it matters
Activation shape affects gradient health, sparsity, and expressivity. Dead ReLUs, exploding sigmoids, and mismatched output activations (e.g., softmax with wrong loss) are frequent training bugs. For LLMs, the LM head almost always pairs logits → softmax with [cross-entropy](/ai-glossary/cross-entropy/) on the next token; the interesting design choice is the MLP activation inside each block.
