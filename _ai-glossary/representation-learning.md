---
title: Representation Learning
related:
  - Embedding
  - Deep Learning
  - Self-Supervised Learning
  - Contrastive Learning
  - Latent Space
  - Transfer Learning
---
**Representation learning** discovers features — [embeddings](/ai-glossary/embedding/), hidden states, or codes — that make downstream tasks easier than working from raw inputs.

### How it’s done
[Self-supervised](/ai-glossary/self-supervised-learning/) objectives (masked/next-token LM, [contrastive learning](/ai-glossary/contrastive-learning/)) and supervised multitask training both produce useful [latent spaces](/ai-glossary/latent-space/). Word embeddings, BERT states, and CLIP-style image–text encoders are classic examples.

### Why practitioners care
Good representations enable [transfer learning](/ai-glossary/transfer-learning/): freeze or lightly tune an encoder and train a small head. In RAG and search, representation quality is retrieval quality — evaluate with task metrics, not only intrinsic loss.
