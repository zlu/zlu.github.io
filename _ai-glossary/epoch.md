---
title: Epoch
related:
  - Batch Size
  - Overfitting
---
An **epoch** is one full pass over the training dataset. LLM continued pretraining is often measured in epochs over your domain corpus *or* in optimizer steps — be explicit which you mean.

More epochs can help fit a domain but raise overfitting and forgetting risk. Track both training loss and held-out domain probes.
