---
title: Hidden State
synonyms:
  - Hidden-State Vector
  - h_t
related:
  - Sequence Model
  - RNN
  - LSTM
  - Encoder
  - Decoder
---
A **hidden state** is the model’s running internal **vector** (a fixed-length list of numbers)—usually written $h_t$—that summarizes “what I’ve processed so far” at step $t$. When people say “the encoder’s state,” they mean that vector, not a discrete label or a text summary.

In an [RNN](/ai-glossary/rnn/) or other [sequence model](/ai-glossary/sequence-model/), each new input $x_t$ is combined with the previous hidden state $h_{t-1}$ to produce $h_t$. That vector is *hidden* in the sense that it is not the final prediction; it is intermediate memory passed along the sequence.

<div class="mermaid">
flowchart LR
  h0["h₀<br/>(empty / zeros)"] --> step1
  x1["x₁: I"] --> step1["combine"]
  step1 --> h1["h₁<br/>seen: I"]
  h1 --> step2
  x2["x₂: love"] --> step2["combine"]
  step2 --> h2["h₂<br/>seen: I love"]
  h2 --> step3
  x3["x₃: coffee"] --> step3["combine"]
  step3 --> h3["h₃<br/>seen: I love coffee"]
</div>

One step in isolation:

<div class="mermaid">
flowchart LR
  htprev["hₜ₋₁<br/>memory so far"] --> f["update<br/>hₜ = f(hₜ₋₁, xₜ)"]
  xt["xₜ<br/>new token"] --> f
  f --> ht["hₜ<br/>updated memory"]
</div>

In early encoder–decoder translation, the encoder’s **final** hidden state (here $h_3$) was treated as a one-shot summary of the whole source sentence—the bottleneck soft attention later removed by keeping *every* position’s state and blending them on demand.
