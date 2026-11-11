---
title: Stochastic Gradient Descent
synonyms:
  - SGD
related:
  - Gradient Descent
  - Optimizer
  - Mini-Batch
  - Learning Rate
  - Adam
---
**Stochastic gradient descent** estimates the gradient on a subset of data (a [mini-batch](/ai-glossary/mini-batch/)) rather than the full dataset each step. That noise scales training to huge corpora and can help escape sharp, poorly generalizing minima.

Classic SGD updates $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta; B)$ for batch $B$ and learning rate $\eta$. Momentum and learning-rate schedules make plain SGD competitive; Adam-family methods ([Adam](/ai-glossary/adam/), AdamW) are more common for LLMs because they adapt per-parameter step sizes.

### Practical notes
- Batch size trades gradient noise for throughput and memory
- Too-large batches can change the loss landscape and need LR retuning
- SGD with momentum still appears in vision and some large-scale LM recipes

SGD is the conceptual parent of almost every modern [optimizer](/ai-glossary/optimizer/); “stochastic” here means sampling data, not sampling weights.
