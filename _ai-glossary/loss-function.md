---
title: Loss Function
synonyms:
  - Objective Function
related:
  - Cross-Entropy
  - Gradient Descent
  - Evaluation
  - Perplexity
---
A **loss function** quantifies prediction error during training. Optimization ([gradient descent](/ai-glossary/gradient-descent/) / Adam-style) minimizes expected loss on the training distribution.

### LLM default
Token-level **[cross-entropy](/ai-glossary/cross-entropy/)** (negative log-likelihood) is standard for next-token prediction. Average CE relates to [perplexity](/ai-glossary/perplexity/) ($\mathrm{PPL} = e^{\overline{\mathrm{CE}}}$). Preference methods ([DPO](/ai-glossary/dpo/)) use different pairwise objectives on top of an LM.

### Loss ≠ product goal
Downstream quality still needs separate task [evaluation](/ai-glossary/evaluation/)—accuracy, groundedness, latency, preference wins. A falling train loss can hide overfitting or format collapse. Log loss curves, but ship on metrics tied to user outcomes.

When comparing runs, match tokenization and packing: loss per token is only comparable under the same data and tokenizer.
