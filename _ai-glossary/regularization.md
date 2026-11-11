---
title: Regularization
related:
  - Dropout
  - Weight Decay
  - Overfitting
  - Early Stopping
---
**Regularization** reduces [overfitting](/ai-glossary/overfitting/) by constraining capacity or injecting noise so the model prefers simpler or more robust solutions.

### Common tools
- **L2 / [weight decay](/ai-glossary/weight-decay/)** — shrink weights; AdamW decouples this from the adaptive step
- **[Dropout](/ai-glossary/dropout/)** — random zeroing of activations in training
- **[Early stopping](/ai-glossary/early-stopping/)** — halt when validation worsens
- Data augmentation, label smoothing, and smaller models / adapters

### LLM practice
For PEFT, limiting LoRA rank and trainable modules is itself regularization. Too much decay or dropout on a small SFT set underfits; too little memorizes. Tune against held-out metrics, not train loss alone.
