---
title: Encoder
synonyms:
  - Source Encoder
related:
  - Decoder
  - Encoder–Decoder
  - Sequence Model
  - Hidden State
  - Sequence-to-Sequence
  - RNN
---
In an encoder–decoder system, the **encoder** reads the *source* sequence (for example the English sentence in translation) and builds representations of it.

In early neural MT the encoder was usually an RNN that walked the source left to right and ended with one final [hidden state](/ai-glossary/hidden-state/). Modern encoder–decoder Transformers instead produce a vector (or stack of vectors) for *every* source position.

The encoder’s job is understanding the input; it does not emit the target tokens. That generation is left to the [decoder](/ai-glossary/decoder/).
