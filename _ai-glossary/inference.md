---
title: Inference
related:
  - Sampling
  - Temperature
  - KV Cache
  - Quantization
---
**Inference** is running a trained model to produce outputs — as opposed to training. For LLMs this means autoregressive decoding with sampling or search strategies.

Latency, throughput, and cost dominate production: batching, KV-cache management, quantization, speculative decoding, and specialized servers (vLLM and friends) matter as much as model quality.
