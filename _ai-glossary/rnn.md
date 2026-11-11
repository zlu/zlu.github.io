---
title: RNN
synonyms:
  - Recurrent Neural Network
related:
  - Sequence Model
  - Hidden State
  - LSTM
  - Encoder
  - Decoder
  - Transformer
---
A **recurrent neural network (RNN)** is the classic [sequence model](/ai-glossary/sequence-model/): it processes tokens step-by-step and passes a [hidden state](/ai-glossary/hidden-state/) through time. Vanilla RNNs struggle with long dependencies and are hard to parallelize across the sequence length.

### Lineage
[LSTMs](/ai-glossary/lstm/)/GRUs improved gating and gradient flow. Early neural MT paired an RNN [encoder](/ai-glossary/encoder/) with an RNN [decoder](/ai-glossary/decoder/), later adding attention. [Transformers](/ai-glossary/transformer/) largely replaced RNNs for large-scale NLP by computing dependencies in parallel via self-attention.

### Today
Recurrence still appears in some efficient or streaming architectures, but production LLMs are overwhelmingly transformer-based. Know RNNs for history and for problems where strict left-to-right state is natural.
