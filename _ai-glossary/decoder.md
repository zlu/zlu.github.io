---
title: Decoder
synonyms:
  - Target Decoder
related:
  - Encoder
  - Encoder–Decoder
  - Decoder-Only
  - Sequence Model
  - Sequence-to-Sequence
  - Causal Language Model
---
In an encoder–decoder system, the **decoder** generates the *target* sequence one step at a time (for example French words), conditioned on what the [encoder](/ai-glossary/encoder/) saw and on tokens already produced.

At each step it forms a prediction for the next token, then feeds that choice back in for the following step. Soft attention lets the decoder *look back* at encoder states instead of trusting a single compressed vector.

**Decoder-only** models (GPT-style) keep the same left-to-right generation habit but drop a separate encoder: the prompt and the answer live in one causal stack.
