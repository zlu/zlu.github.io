---
title: Masked Model
description: "A masked model (masked language model, MLM) recovers hidden tokens using left and right context. BERT-style encoders; contrast with causal LMs."
keywords: "masked model, maskedmodel, masked language model, MLM, BERT, bidirectional encoder"
synonyms:
  - Masked Language Model
  - MLM
related:
  - Causal Model
  - Causal Language Model
  - Self-Supervised Learning
  - Encoder
  - Token
---
A **masked model** (masked language model, MLM) is trained to recover deliberately hidden spans of the input using bidirectional context. BERT-style [encoders](/ai-glossary/encoder/) replace random [tokens](/ai-glossary/token/) with a mask token and predict the originals. Classic [self-supervised](/ai-glossary/self-supervised-learning/) pretraining.

### MLM vs causal LM

| | Masked model (MLM) | Causal LM |
| --- | --- | --- |
| Context | Left and right | Left only |
| Typical stack | Encoder (BERT-style) | Decoder-only (GPT-style) |
| Training target | Recover masked positions | Next-token prediction |
| Strength | NLU, embeddings, NER | Open-ended generation |
| Generation | Not the default use | Native left-to-right text |

See [causal language model](/ai-glossary/causal-language-model/) for the unidirectional case most chat LLMs use.

### Why it matters
- Strong for understanding tasks: classification, NER, embeddings
- Sees left *and* right context, unlike causal LMs
- Less natural for open-ended left-to-right generation

### How it shows up
Training objective: minimize cross-entropy on masked positions only. At inference, you typically fine-tune or pool hidden states rather than "unmask" free-form prose. Modern LLMs are mostly [decoder-only](/ai-glossary/decoder-only/) / causal; masked (and encoder-decoder) models still matter for retrieval encoders, rerankers, and classical NLU pipelines.

Related: span corruption (T5) is a close cousin that corrupts contiguous spans instead of single tokens.
