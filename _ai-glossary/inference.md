---
title: Inference
related:
  - Sampling
  - Temperature
  - KV Cache
  - Quantization
  - Speculative Decoding
---
**Inference** is running a trained model to produce outputs—as opposed to training. For LLMs this means autoregressive [decoding](/ai-glossary/decoding/) with [sampling](/ai-glossary/sampling/) or search strategies ([temperature](/ai-glossary/temperature/), top-p, beam search, greedy).

### Production bottlenecks
Latency, throughput, and cost often dominate model quality debates. Batching, [KV-cache](/ai-glossary/kv-cache/) management, [quantization](/ai-glossary/quantization/), [speculative decoding](/ai-glossary/speculative-decoding/), and specialized servers (vLLM and friends) matter as much as the checkpoint.

### Train vs serve
Training optimizes weights under a loss; inference chooses tokens under decoding hyperparameters and hardware limits. A great SFT model can still fail SLOs if context is long, batching is poor, or KV memory thrashs GPUs.

Document decoding settings alongside eval results—apples-to-apples comparisons require identical inference configs.
