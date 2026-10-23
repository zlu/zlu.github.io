---
title: Sequence-to-Sequence
synonyms:
  - Seq2Seq
related:
  - Encoder–Decoder
  - Encoder
  - Decoder
  - Sequence Model
  - Attention
  - Transformer
---
**Sequence-to-sequence** (seq2seq) models map an input sequence to an output sequence of possibly different length—translation, summarization, speech recognition.

The usual architecture is [encoder–decoder](/ai-glossary/encoder-decoder/): read the source, then generate the target. Soft [attention](/ai-glossary/attention/) made long sources workable; Transformers later replaced most RNN seq2seq stacks. Decoder-only models can also do seq2seq via prompting or special formats.

