---
title: Optimizer
related:
  - Adam
  - Gradient Descent
  - Stochastic Gradient Descent
  - Learning Rate
  - Weight Decay
  - Warmup
---
An **optimizer** turns gradients into parameter updates. Basics: [gradient descent](/ai-glossary/gradient-descent/) and [SGD](/ai-glossary/stochastic-gradient-descent/) (with optional momentum). Adaptive methods ([Adam](/ai-glossary/adam/), AdamW) track per-parameter statistics.

### Defaults for LLMs
**AdamW** (Adam + decoupled [weight decay](/ai-glossary/weight-decay/)) is the usual choice for transformers. Pair with a sensible [learning rate](/ai-glossary/learning-rate/), [warmup](/ai-glossary/warmup/), and schedule (cosine / constant). Tuning LR, decay, and batch size together matters more than swapping exotic optimizers.

### What to watch
Wrong LR or missing warmup → loss spikes. Too little weight decay → overfitting on small SFT sets. Report effective tokens/step when comparing runs.
