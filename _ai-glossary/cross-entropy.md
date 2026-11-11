---
title: Cross-Entropy
related:
  - Loss Function
  - Softmax
  - Perplexity
---
**Cross-entropy** measures how well a predicted probability distribution matches a target distribution. As a [loss](/ai-glossary/loss-function/), classification compares softmax probabilities with a one-hot (or soft) label; language modeling uses the negative log probability of the correct next [token](/ai-glossary/token/).

### Why it matters
- Default training objective for classifiers and LMs
- [Perplexity](/ai-glossary/perplexity/) is $\exp$ of average cross-entropy—same information, friendlier scale
- Numerically implemented as log-softmax + NLL for stability, not naive softmax then log

### How it shows up
LM head: logits → ([softmax](/ai-glossary/softmax/) implicitly) → CE on the gold token. Soft labels / label smoothing change the target distribution. Don’t confuse training CE with decoding algorithms—low CE means calibrated next-token probs on the train/eval distribution, not that [beam search](/ai-glossary/beam-search/) or sampling will “sound better” for every product task.
