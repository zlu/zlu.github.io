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
---
An **MLP** (**multi-layer perceptron**) is a small stack of fully connected layers: each layer is a linear map (matrix × vector + bias) followed by a nonlinear [activation](/ai-glossary/activation-function/).

In the soft-attention score of Bahdanau et al., a tiny MLP takes the concatenated query and encoder state $ [q; h_i] $ and outputs a single compatibility number. In Transformers, the position-wise [feed-forward network](/ai-glossary/feed-forward-network/) inside each block is also an MLP (usually two linear layers with a nonlinearity in between).
