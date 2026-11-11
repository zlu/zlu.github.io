---
title: MLP
synonyms:
  - Multi-Layer Perceptron
  - Multilayer Perceptron
related:
  - Feed-Forward Network
  - Neural Network
  - Activation Function
  - Softmax
  - Transformer
---
An **MLP** (**multi-layer perceptron**) is a stack of fully connected layers: each layer is a linear map (matrix × vector + bias) followed by a nonlinear [activation](/ai-glossary/activation-function/).

### In transformers
The position-wise [feed-forward network](/ai-glossary/feed-forward-network/) in each block is an MLP (usually expand → nonlinearity → project back). Modern LLMs often use GELU or SwiGLU rather than ReLU. Attention mixes tokens; the MLP transforms each position independently and holds most of the dense parameter count.

### Elsewhere
Small MLPs appear as scoring heads (e.g., Bahdanau attention over $[q; h_i]$), classification heads, and router networks in MoE. “MLP” and “FFN” are often used interchangeably in transformer jargon.
