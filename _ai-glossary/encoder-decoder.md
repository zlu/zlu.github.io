---
title: Encoder–Decoder
synonyms:
  - Seq2Seq Architecture
  - Encoder-Decoder
related:
  - Encoder
  - Decoder
  - Sequence-to-Sequence
  - Sequence Model
  - Attention
  - Transformer
  - Decoder-Only
---
An **encoder–decoder** model splits sequence work in two:

1. The [encoder](/ai-glossary/encoder/) reads the source and builds representations.
2. The [decoder](/ai-glossary/decoder/) writes the target, step by step, using those representations.

Classic neural machine translation used RNN encoders and decoders with a *single* fixed vector between them—the famous bottleneck that soft attention removed. Transformer encoder–decoder models (T5, BART, and many MT systems) keep the split but pass a full sequence of encoder states to the decoder via cross-attention.

Many chat LLMs instead use **decoder-only** stacks for simplicity and scaling; the encoder–decoder pattern remains strong for translation, summarization, and other clear source→target maps.
