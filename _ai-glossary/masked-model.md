---
title: Masked Model
synonyms:
  - Masked Language Model
  - MLM
related:
  - Causal Model
  - Causal Language Model
  - Self-Supervised Learning
  - Encoder
---
A **masked model** (masked language model, MLM) is trained to recover deliberately hidden spans of the input using bidirectional context. BERT-style [encoders](/ai-glossary/encoder/) replace random [tokens](/ai-glossary/token/) with a mask token and predict the originals—classic [self-supervised](/ai-glossary/self-supervised-learning/) pretraining.

### Why it matters
- Strong for understanding tasks: classification, NER, embeddings
- Sees left *and* right context, unlike [causal LMs](/ai-glossary/causal-language-model/)
- Less natural for open-ended left-to-right generation

### How it shows up
Training objective: minimize cross-entropy on masked positions only. At inference, you typically fine-tune or pool hidden states rather than “unmask” free-form prose. Modern LLMs are mostly [decoder-only](/ai-glossary/decoder-only/) / causal; masked (and encoder–decoder) models still matter for retrieval encoders, rerankers, and classical NLU pipelines.

Related: span corruption (T5) is a close cousin that corrupts contiguous spans instead of single tokens.
