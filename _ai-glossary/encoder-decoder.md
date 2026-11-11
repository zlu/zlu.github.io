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
  - Self-Attention
---
An **encoder–decoder** model splits sequence work in two:

1. The [encoder](/ai-glossary/encoder/) reads the source and builds representations (often bidirectional).
2. The [decoder](/ai-glossary/decoder/) writes the target, step by step, using those representations via cross-[attention](/ai-glossary/attention/).

Classic neural machine translation used RNN encoders and decoders with a *single* fixed vector between them—the famous bottleneck that soft attention removed. [Transformer](/ai-glossary/transformer/) encoder–decoder models (T5, BART, and many MT systems) keep the split but pass a full sequence of encoder states to the decoder.

Many chat LLMs instead use **decoder-only** stacks for simplicity and scaling; the encoder–decoder pattern remains strong for translation, summarization, and other clear source→target maps. Training typically pairs [teacher forcing](/ai-glossary/teacher-forcing/) on the target with full source context in the encoder.

See also [sequence-to-sequence](/ai-glossary/sequence-to-sequence/) for the task framing this architecture was built to solve.
