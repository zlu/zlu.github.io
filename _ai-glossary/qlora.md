---
title: QLoRA
synonyms:
  - Quantized LoRA
related:
  - LoRA
  - Quantization
  - PEFT
---
**QLoRA** combines 4-bit (or similar) quantization of the base model with LoRA adapters trained in higher precision. The frozen base stays memory-cheap; gradients flow through the adapters.

It is popular for fine-tuning 7B–70B models on a single consumer or mid-tier GPU. Quality depends on quantization scheme, LoRA capacity, and data — not magic by itself.
