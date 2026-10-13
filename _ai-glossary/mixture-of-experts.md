---
title: Mixture of Experts
synonyms:
  - MoE
related:
  - Feed-Forward Network
  - Transformer
  - Inference
---
**Mixture of Experts (MoE)** routes each token to a subset of expert FFNs, increasing parameter count without proportionally increasing compute per token.

MoE models can be very capable per FLOP but need careful load balancing and serving infrastructure.
