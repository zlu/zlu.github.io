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
  - Teacher Forcing
---
**Sequence-to-sequence** (seq2seq) models map an input sequence to an output sequence of possibly different length—translation, summarization, speech recognition, and many structured generation tasks.

The usual architecture is [encoder–decoder](/ai-glossary/encoder-decoder/): the [encoder](/ai-glossary/encoder/) reads the source; the [decoder](/ai-glossary/decoder/) generates the target one token at a time. Soft [attention](/ai-glossary/attention/) made long sources workable by letting the decoder look back at every encoder state instead of one bottleneck vector. Transformers later replaced most RNN seq2seq stacks while keeping the same split.

Training often uses [teacher forcing](/ai-glossary/teacher-forcing/) (feed gold previous tokens). At inference the decoder consumes its own outputs. Decoder-only LLMs can also do seq2seq via prompting or special formats (prefix the source, generate the target), but a dedicated encoder–decoder remains natural when the source and target roles are clear and asymmetric.
