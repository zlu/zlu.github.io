---
title: Tokenizer
related:
  - Token
  - Embedding
  - Byte-Pair Encoding
---
A **tokenizer** converts raw text into a sequence of [token](/ai-glossary/token/) IDs the model was trained with (and back). Subword schemes ([BPE](/ai-glossary/byte-pair-encoding/), WordPiece, Unigram) dominate modern LLMs; character- and byte-level schemes appear in some multilingual and robustness-focused designs.

Tokenizer choice affects fertility (tokens per word), multilingual fairness, code handling, and effective context length. A word that splits into many pieces costs more context and can hurt modeling.

### Practical rules
- Always use the tokenizer that matches the checkpoint
- Watch special tokens (BOS/EOS, chat markers, tool delimiters)
- Do not mix vocabularies across models when concatenating embeddings or logits

Detokenization must reverse the same rules—spacing and byte fallbacks differ across libraries. Mismatched tokenize/detokenize pairs are a common source of “mystery” eval bugs.
