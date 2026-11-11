---
title: Dropout
related:
  - Regularization
  - Overfitting
  - Fine-Tuning
---
**Dropout** randomly zeroes activations during training so the network cannot rely on brittle co-adaptations. At evaluation time, dropout is disabled (or inverted dropout scales at train time) so inference uses the full network.

### Where it shows up
Transformers often apply modest dropout on [attention](/ai-glossary/attention/) weights, residual streams, and [FFN](/ai-glossary/feed-forward-network/) blocks. [LoRA](/ai-glossary/lora/) adapters also expose a dropout knob on the low-rank path.

### Practical notes
- Fine-tuning small datasets sometimes *lowers* dropout to fit faster; large noisy corpora may keep or raise it
- Too much dropout underfits; too little invites [overfitting](/ai-glossary/overfitting/)
- Pair with early stopping and held-out eval—dropout alone is not a complete regularizer

Modern LLMs pretrained at scale often use light dropout; treat it as a secondary knob after learning rate and data quality.
