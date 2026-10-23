---
title: Sequence Model
synonyms:
  - Sequential Model
  - Sequence Model (RNN-style)
related:
  - Hidden State
  - RNN
  - Sequence-to-Sequence
  - Encoder
  - Decoder
  - LSTM
---
A **sequence model** processes ordered inputs one step at a time (or in a stack that respects order) and keeps a running summary of what it has seen so far—the [*hidden state*](/ai-glossary/hidden-state/).

Classic examples are RNNs, LSTMs, and GRUs reading tokens left to right. The same idea appears in speech and time series: at step $t$, the model combines the new observation $x_t$ with the previous summary $h_{t-1}$ to form $h_t$.

That running summary is powerful for short context and fragile for long context: early information can be overwritten. Soft attention and Transformers were invented largely to escape that pressure.
