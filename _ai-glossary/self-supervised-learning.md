---
title: Self-Supervised Learning
synonyms:
  - SSL
related:
  - Pretraining
  - Unsupervised Learning
  - Contrastive Learning
  - Next-Token Prediction
  - Masked Model
---
**Self-supervised learning (SSL)** creates training signals from the input itself — predict masked spans, [next tokens](/ai-glossary/next-token-prediction/), or agreement across augmented views — so you can [pretrain](/ai-glossary/pretraining/) at web scale without hand labels.

### Vs unsupervised
Often grouped with [unsupervised learning](/ai-glossary/unsupervised-learning/), but SSL has an explicit pretext loss (cross-entropy, contrastive, etc.), not only clustering or density estimation. [Contrastive learning](/ai-glossary/contrastive-learning/) and masked/[causal](/ai-glossary/causal-language-model/) LM are the dominant recipes.

### Why it matters
Modern foundation models are primarily self-supervised learners; supervised SFT and preference data are comparatively small polish stages. Choose the pretext to match the modality and downstream use.
