---
title: DPO
synonyms:
  - Direct Preference Optimization
related:
  - RLHF
  - Alignment
  - Supervised Fine-Tuning
---
**Direct Preference Optimization (DPO)** aligns a model to preference pairs (chosen vs rejected responses) without training a separate reward model or running PPO.

It reparameterizes the RLHF objective into a classification-style loss on the policy. DPO is widely used as a lighter alternative or complement to classic RLHF.
