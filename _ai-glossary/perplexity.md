---
title: Perplexity
synonyms:
  - PPL
related:
  - Cross-Entropy
  - Causal Language Model
  - Evaluation
  - Next-Token Prediction
---
**Perplexity (PPL)** is $\exp(\text{average negative log-likelihood})$ over tokens — equivalently $\exp(\text{cross-entropy})$ in nats. Lower means the [causal LM](/ai-glossary/causal-language-model/) is less “surprised” by the text under [next-token prediction](/ai-glossary/next-token-prediction/).

### What it’s good for
Comparing language models on the **same** tokenizer, domain, and preprocessing. Tracking training progress and domain fit after CPT.

### What it isn’t
PPL does not measure helpfulness, factuality, safety, or instruction-following. Tokenization differences make cross-family comparisons misleading. Pair PPL with task [evaluation](/ai-glossary/evaluation/) (benchmarks, human prefs, RAG faithfulness) for product decisions.
