---
title: Unsupervised Learning
related:
  - Supervised Learning
  - Self-Supervised Learning
  - Representation Learning
---
**Unsupervised learning** finds structure in unlabeled data—clustering, density estimation, dimensionality reduction—without explicit target labels $y$.

Classical examples include $k$-means, PCA, and topic models. The learner organizes $x$ by similarity or latent factors rather than predicting a provided answer.

### Relation to modern LMs
- [Self-supervised learning](/ai-glossary/self-supervised-learning/) invents pretext labels from the data (next token, masked token)—not the same as classical unsupervised, though both avoid human labels
- [Representation learning](/ai-glossary/representation-learning/) often starts unsupervised or self-supervised, then transfers
- Pure unsupervised clustering still appears in analytics and as a diagnostic on embeddings

When someone says an LLM is “unsupervised,” they usually mean self-supervised pretraining on raw text. Prefer the precise term when writing docs or papers.
