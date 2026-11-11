---
title: Underfitting
related:
  - Overfitting
  - Regularization
  - Dropout
  - Learning Rate
  - LoRA
---
**Underfitting** means the model is too weak or under-trained to capture the signal—both train and validation metrics stay poor. Unlike [overfitting](/ai-glossary/overfitting/), the gap between train and val is small; everything is just bad.

### Common causes
- Capacity too low (tiny model, tiny [LoRA](/ai-glossary/lora/) rank)
- Training too short or [learning rate](/ai-glossary/learning-rate/) too low
- Excessive [regularization](/ai-glossary/regularization/) / [dropout](/ai-glossary/dropout/) / weight decay
- Features or labels that do not contain the signal
- Mismatched objective (wrong loss or chat template)

### Fixes
- More capacity or higher adapter rank
- Longer training / better LR schedule
- Weaker regularizers
- Better data, labels, or preprocessing

If train loss is already near zero but val is terrible, you are overfitting—not underfitting.
