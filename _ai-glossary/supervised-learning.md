---
title: Supervised Learning
related:
  - Unsupervised Learning
  - Self-Supervised Learning
  - Independent Variable
  - Dependent Variable
  - Supervised Fine-Tuning
  - Loss Function
---
**Supervised learning** trains on labeled examples $(x, y)$ to predict $y$ from $x$—classification, regression, or sequence labeling. The [loss](/ai-glossary/loss-function/) compares predictions to the provided targets.

Classic examples: image classifiers with human labels, spam filters, named-entity taggers. In the LLM world, [supervised fine-tuning](/ai-glossary/supervised-fine-tuning/) is supervised learning on instruction–response pairs; next-token [pretraining](/ai-glossary/pretraining/) is better described as [self-supervised](/ai-glossary/self-supervised-learning/) because labels are derived from the text itself.

### Contrasts
- [Unsupervised](/ai-glossary/unsupervised-learning/) — no explicit $y$; find structure in $x$
- Self-supervised — invent $y$ from $x$ (mask a token, predict the next)
- Reinforcement learning — learn from rewards, not fixed labels

Quality of labels and train/validation splits dominate outcomes; fancy architectures rarely fix a noisy or leaky dataset.
