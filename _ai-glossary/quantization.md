---
title: Quantization
related:
  - Inference
  - Mixed Precision
  - GGUF
  - QLoRA
  - Fine-Tuning
---
**Quantization** reduces the numeric precision of model weights (and sometimes activations)—e.g., float16/bfloat16 down to int8, int4, or other low-bit formats—to cut memory and often speed up [inference](/ai-glossary/inference/).

### Why people quantize
- Fit larger models on smaller GPUs / CPUs / edge devices
- Higher tokens/sec when kernels are memory-bandwidth bound
- Cheaper multi-tenant serving for the same hardware budget

### Common flavors
- **Post-training quantization (PTQ)** — compress a finished checkpoint; optional calibration data
- **Quantization-aware training (QAT)** — simulate low precision during training
- **Weight-only vs weights+activations** — activations are harder; many LLM stacks quantize weights first

Formats and toolchains include [GGUF](/ai-glossary/gguf/)/llama.cpp, bitsandbytes, GPTQ/AWQ, and others. [QLoRA](/ai-glossary/qlora/) fine-tunes adapters on a quantized base. Expect some quality loss at aggressive bit-widths; measure task metrics, not only perplexity.
