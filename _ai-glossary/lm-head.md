---
title: LM Head
synonyms:
  - Final Vocab Projection
  - Vocabulary Projection
  - Output Projection
  - Unembedding
related:
  - Embedding
  - Logit
  - Softmax
  - Causal Language Model
  - Next-Token Prediction
---
The **LM head** (final vocab projection) is the last linear layer of a [causal language model](/ai-glossary/causal-language-model/). It maps each position’s hidden vector to a vector of [logits](/ai-glossary/logit/), one score per vocabulary token. Softmax (or log-softmax inside [cross-entropy](/ai-glossary/cross-entropy/)) turns those logits into next-token probabilities for [next-token prediction](/ai-glossary/next-token-prediction/).

### Weight tying
Some models **tie** this matrix to the input [embedding](/ai-glossary/embedding/) table (same parameters, transposed). Others learn a separate projection. Either way, pretraining updates it so the true next token’s logit rises relative to the rest of the vocabulary.

### Fine-tuning notes
PEFT setups sometimes freeze the LM head; domain vocab shifts or special tokens may need it trainable. At inference, decoding reads these logits every step—temperature and top-p act on them before sampling.
