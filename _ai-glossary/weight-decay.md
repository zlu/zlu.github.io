---
title: Weight Decay
related:
  - Regularization
  - Adam
  - Overfitting
  - Optimizer
  - Learning Rate
---
**Weight decay** penalizes large weights (L2-style), encouraging simpler solutions and reducing [overfitting](/ai-glossary/overfitting/). Conceptually it pulls parameters toward zero each step; the exact implementation differs across optimizers.

In AdamW, weight decay is applied *separately* from the adaptive gradient update—unlike naive L2 loss added inside Adam, which interacts badly with the adaptive scaling. That decoupling is why AdamW became the default for transformers.

### Practical notes
- Standard stabilizer for LLM pretraining and fine-tuning
- Tune jointly with [learning rate](/ai-glossary/learning-rate/); they trade off
- Often disabled or reduced on biases, LayerNorm, and embeddings
- Too strong → [underfitting](/ai-glossary/underfitting/); too weak → overfit small SFT sets

Weight decay is a form of [regularization](/ai-glossary/regularization/), complementary to dropout and early stopping rather than a replacement.
