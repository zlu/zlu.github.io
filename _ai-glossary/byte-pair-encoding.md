---
title: Byte-Pair Encoding
synonyms:
  - BPE
related:
  - Tokenizer
  - Token
---
**Byte-Pair Encoding (BPE)** builds a subword vocabulary by iteratively merging the most frequent symbol pairs, starting from characters or bytes.

It balances vocabulary size against sequence length and handles rare words by splitting them. Many LLM tokenizers are BPE or BPE-like (including byte-level variants).
