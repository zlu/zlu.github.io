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
A **recurrent neural network** is the classic [sequence model](/ai-glossary/sequence-model/): it processes tokens step-by-step and passes a [hidden state](/ai-glossary/hidden-state/) through time. Vanilla RNNs struggle with long dependencies and are hard to parallelize.

LSTMs/GRUs improved the gating. Early neural MT paired an RNN [encoder](/ai-glossary/encoder/) with an RNN [decoder](/ai-glossary/decoder/). Transformers largely replaced RNNs for large-scale NLP, though recurrence still appears in some efficient architectures.

