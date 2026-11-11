---
title: Byte-Pair Encoding
synonyms:
  - BPE
related:
  - Tokenizer
  - Token
---
**Byte-Pair Encoding (BPE)** builds a subword vocabulary by iteratively merging the most frequent adjacent symbol pairs, starting from characters or bytes. The learned merge table becomes the [tokenizer](/ai-glossary/tokenizer/) used at train and serve time.

### Why it matters
- Balances vocabulary size against sequence length
- Handles rare and morphologically rich words by splitting them
- Most LLM tokenizers are BPE or BPE-like (including byte-level variants)

### How it shows up
Token count ≠ word count: code, non-English text, and numbers often tokenize inefficiently and burn [context window](/ai-glossary/context-window/). Never mix tokenizers across models—embedding and LM head rows are tied to a specific vocab. When debugging “weird” generations, inspect the actual [token](/ai-glossary/token/) IDs, not just the decoded string.
