---
title: QLoRA
synonyms:
  - Quantized LoRA
related:
  - LoRA
  - Quantization
  - PEFT
  - Fine-Tuning
  - Mixed Precision
---
**QLoRA** fine-tunes [LoRA](/ai-glossary/lora/) adapters on a base model stored in low-bit [quantization](/ai-glossary/quantization/) (commonly 4-bit, e.g. NF4). The frozen base stays memory-cheap; adapters and optimizer states use higher precision so gradients remain usable.

### Why it exists
Full fine-tunes of 7B–70B models need a lot of VRAM. QLoRA makes [PEFT](/ai-glossary/peft/) feasible on a single consumer or mid-tier GPU by stacking quantization + LoRA (plus paging tricks in the original recipe).

### Knobs
Quantization type/bits, LoRA rank and target modules, and data quality dominate outcomes. Expect some quality gap vs full-precision LoRA; measure on your eval set rather than assuming “4-bit is free.”
