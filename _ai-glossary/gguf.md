---
title: GGUF
related:
  - Quantization
  - Inference
  - Open-Source Model
---
**GGUF** is a file format commonly used with llama.cpp-style runtimes to ship quantized LLM weights for local CPU/GPU [inference](/ai-glossary/inference/). It packages tensors plus metadata (architecture, tokenizer info, quant type) for portable desktop and edge deployment.

### Why it exists
Earlier ad-hoc GGML dumps were brittle across versions. GGUF standardizes layout so one `.gguf` file can move between tools that speak the format, with clear quant tags (e.g., Q4_K_M, Q8_0).

### Practical notes
- Smaller quants save RAM/disk but can hurt quality—benchmark your tasks
- Match the runtime build to GPU offload, Metal, CUDA, etc.
- Still [quantization](/ai-glossary/quantization/) of a trained model; it is not a training checkpoint format like full safetensors for continued fine-tuning

For local open models, GGUF + llama.cpp (or forks) is a default path from Hugging Face weights to a laptop binary.
