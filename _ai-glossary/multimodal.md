---
title: Multimodal
synonyms:
  - Multi-Modal
related:
  - Embedding
  - Transformer
  - Embedding Model
  - RAG
  - Token
---
**Multimodal** models accept or produce more than one modality — text, images, audio, video. Typical designs encode each modality into a shared or aligned space, then fuse with a [transformer](/ai-glossary/transformer/) (cross-attention, early fusion, or discrete multimodal [tokens](/ai-glossary/token/)).

### Practical notes
- Vision–language models often freeze or lightly tune a vision encoder and project into the LLM’s embedding space
- Evaluate each modality’s failure modes separately (OCR, resolution, speech noise)
- Latency and context cost grow with image/audio patches or frames

[RAG](/ai-glossary/rag/) and agents increasingly mix text retrieval with vision/audio inputs; grounding still depends on retrieval quality and how the prompt presents non-text evidence.
