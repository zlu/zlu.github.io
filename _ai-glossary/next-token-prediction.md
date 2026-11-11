---
title: Next-Token Prediction
synonyms:
  - Autoregressive Language Modeling
related:
  - Causal Language Model
  - Pretraining
  - Cross-Entropy
  - Autoregressive
  - Teacher Forcing
---
**Next-token prediction** trains a model to estimate $P(x_t \mid x_{<t})$ for each position. The training loss is usually average [cross-entropy](/ai-glossary/cross-entropy/) (negative log-likelihood) under [teacher forcing](/ai-glossary/teacher-forcing/).

### Role in LLMs
It is the core [pretraining](/ai-glossary/pretraining/) (and continued-pretraining) objective for [causal](/ai-glossary/causal-language-model/) / [autoregressive](/ai-glossary/autoregressive/) LLMs. At scale it induces syntax, facts, and in-context learning — even though the label at each step is just “the next token.”

### Caveats
The objective does not directly optimize helpfulness, truthfulness, or tool use; those come from SFT and preference/RL stages. Metrics like perplexity track NTP quality but not product quality alone.
