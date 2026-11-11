---
title: Latent Space
related:
  - Embedding
  - Representation Learning
  - Embedding Model
---
A **latent space** is the continuous representation space where a model encodes inputs. Geometry in this space—clusters, directions, neighborhoods—often corresponds to semantic factors the model has learned via [representation learning](/ai-glossary/representation-learning/).

### In practice
[Embeddings](/ai-glossary/embedding/) for retrieval live in a latent space trained so that similarity matches task notions of relatedness. Generative models also maintain residual-stream “latents” per token; probing and steering research studies directions in those spaces.

### Caveats
Latent axes are rarely human-labeled by default. Changing the [embedding model](/ai-glossary/embedding-model/), pooling, or fine-tuning moves the space—indexes and classifiers must be rebuilt. Visualize with care: 2D projections can invent structure that is not there at full dimensionality.

For [RAG](/ai-glossary/rag/), treat the embedding latent space as a versioned contract: same model, same preprocessing, same metric (cosine vs L2).
