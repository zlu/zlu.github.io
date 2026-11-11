---
title: Epoch
related:
  - Batch Size
  - Overfitting
  - Learning Rate
---
An **epoch** is one full pass over the training dataset. LLM continued pretraining is often measured in epochs over your domain corpus *or* in optimizer steps—be explicit which you mean when comparing runs.

### Why the distinction matters
With huge corpora you may never finish an epoch; schedules and [warmup](/ai-glossary/warmup/) are step-based. On small SFT sets, multiple epochs are normal, but repeating the same few thousand examples raises [overfitting](/ai-glossary/overfitting/) and forgetting risk.

### Practical guidance
- Track training loss *and* held-out domain probes each epoch or every $N$ steps
- More epochs can help fit a domain; stop when validation plateaus ([early stopping](/ai-glossary/early-stopping/))
- Effective exposure also depends on [batch size](/ai-glossary/batch-size/) and data packing—token count often beats “epochs” as a fair unit

Report tokens seen, not only epoch count, when sharing fine-tuning recipes.
