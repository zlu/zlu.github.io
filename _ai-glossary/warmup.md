---
title: Warmup
related:
  - Learning Rate
  - Optimizer
  - Adam
  - Fine-Tuning
---
**Learning-rate warmup** starts training with a smaller [learning rate](/ai-glossary/learning-rate/) and ramps up over the first $N$ steps before the main decay schedule (linear, cosine, etc.).

It stabilizes early updates when gradients or adaptive [optimizer](/ai-glossary/optimizer/) statistics ([Adam](/ai-glossary/adam/) moments) are noisy—common practice in transformer pretraining and [fine-tuning](/ai-glossary/fine-tuning/).

### Why early steps are fragile
- Random init or freshly attached heads produce large, poorly scaled gradients
- Adaptive optimizers have unreliable second-moment estimates until enough steps accumulate
- A full LR from step 0 can spike loss or destroy pretrained features

Typical recipes: linear warmup for a few hundred to a few thousand steps, then cosine decay. Warmup length should scale with total steps and batch size; too long wastes training, too short reintroduces early instability.
