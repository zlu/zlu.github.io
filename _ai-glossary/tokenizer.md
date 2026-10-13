---
title: Tokenizer
related:
  - Token
  - Embedding
  - Byte-Pair Encoding
---
A **tokenizer** converts raw text into a sequence of token IDs the model was trained with (and back). Subword schemes (BPE, WordPiece, Unigram) dominate modern LLMs.

Tokenizer choice affects fertility (tokens per word), multilingual fairness, and code handling. Always use the tokenizer that matches the checkpoint.
