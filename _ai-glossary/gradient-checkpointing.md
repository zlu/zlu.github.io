---
title: Gradient Checkpointing
synonyms:
  - Activation Checkpointing
related:
  - Mixed Precision
  - Backpropagation
  - Fine-Tuning
---
**Gradient checkpointing** saves memory by not storing all intermediate activations: during [backward](/ai-glossary/backpropagation/), selected layers are recomputed from saved checkpoints. You trade extra compute for the ability to train longer sequences or larger micro-batches.

### When to use it
- Long-context [LoRA](/ai-glossary/lora/) / [CPT](/ai-glossary/continued-pretraining/) on a single GPU
- Full fine-tunes that OOM even with [mixed precision](/ai-glossary/mixed-precision/)
- Stacked with [gradient accumulation](/ai-glossary/gradient-accumulation/) when both sequence length and effective batch matter

### Cost
Expect slower steptime (often ~20–30%+ depending on what you recompute). Prefer checkpointing transformer blocks rather than every tiny op. It does not change the mathematical gradient—only the memory/time tradeoff.

Enable it when you are VRAM-bound; leave it off for short-context runs where throughput matters more than memory headroom.
