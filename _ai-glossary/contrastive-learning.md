---
title: Contrastive Learning
related:
  - Embedding
  - Self-Supervised Learning
  - Cosine Similarity
  - Dense Retrieval
---
**Contrastive learning** trains encoders so related pairs (augmentations, dual views, text–image matches) land close in [embedding](/ai-glossary/embedding/) space and unrelated pairs stay far apart—often measured with [cosine similarity](/ai-glossary/cosine-similarity/).

### Why it matters
- Backbone of CLIP-style multimodal models and many retrieval embedding models
- InfoNCE / NT-Xent-style losses make batch negatives do useful work
- Embedding quality for [dense retrieval](/ai-glossary/dense-retrieval/) hinges on positives, hard negatives, and batch size

### How it shows up
In-batch negatives are free but biased; mined hard negatives improve ranking. Temperature and normalization in the loss are high-leverage knobs. At serve time you usually freeze the encoder and run ANN search—training is contrastive; inference is nearest-neighbor lookup.
