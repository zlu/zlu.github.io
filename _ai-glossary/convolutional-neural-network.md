---
title: Convolutional Neural Network
synonyms:
  - CNN
related:
  - Deep Learning
  - Neural Network
  - Transformer
---
A **CNN** applies shared local filters (convolutions) across space or time, building hierarchical feature maps with translation-friendly inductive bias. Pooling and striding control resolution; depth stacks local patterns into higher-level concepts.

### Why it matters
- Long-dominant architecture for vision; still strong and efficient on many perceptual tasks
- Fewer parameters than dense layers for the same local receptive field
- Vision Transformers compete; hybrids and conv stems remain common

### How it shows up
Classic stacks: conv → norm → activation → pool, repeated. Outside vision, 1-D convolutions appear in speech and some efficient LM variants. For LLM practitioners, CNNs are less central than [Transformers](/ai-glossary/transformer/), but the same ideas—weight sharing, local connectivity, hierarchical features—still inform efficient architectures.
