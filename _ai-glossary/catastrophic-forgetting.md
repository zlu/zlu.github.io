---
title: Catastrophic Forgetting
related:
  - Continued Pretraining
  - Fine-Tuning
  - Transfer Learning
  - LoRA
---
**Catastrophic forgetting** is when training on a new distribution erases previously learned skills. Domain [continued pretraining](/ai-glossary/continued-pretraining/) or aggressive [fine-tuning](/ai-glossary/fine-tuning/) can boost in-domain fluency while quietly tanking general benchmarks or unrelated tasks.

### Why it matters
- Domain LLM recipes routinely trade generality for specialization
- Undetected unless you keep out-of-domain probes in the eval suite
- Full-weight updates forget faster than small [adapters](/ai-glossary/adapter/) at the same aggressiveness

### Mitigations
Lower [learning rates](/ai-glossary/learning-rate/), prefer [LoRA](/ai-glossary/lora/)/PEFT over full FT when possible, mix replay of general data, [early stopping](/ai-glossary/early-stopping/), and evaluate both in-domain and held-out general tasks. Forgetting is a spectrum—measure it; don’t assume a single [epoch](/ai-glossary/epoch/) is “safe.”
