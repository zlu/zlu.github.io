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
  - Attention
  - Transformer
---
In an encoder–decoder system, the **encoder** reads the *source* sequence (for example the English sentence in translation) and builds representations of it.

In early neural MT the encoder was usually an [RNN](/ai-glossary/rnn/) that walked the source left to right and ended with one final [hidden state](/ai-glossary/hidden-state/). Soft [attention](/ai-glossary/attention/) already needed every source state; modern encoder–decoder [Transformers](/ai-glossary/transformer/) produce a vector (or stack of vectors) for *every* source position via bidirectional [self-attention](/ai-glossary/self-attention/).

The encoder’s job is understanding the input; it does not emit the target tokens. That generation is left to the [decoder](/ai-glossary/decoder/). Encoder-only stacks (BERT-style) reuse the same idea for classification and span tasks without a generative decoder.
