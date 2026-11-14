---
title: Vision-Language Model
title_short: VLM
synonyms:
  - VLM
  - Vision Language Model
related:
  - Multimodal
  - Token
  - Instruct Model
  - Inference
  - Zero-Shot
---
A **vision-language model (VLM)** takes an image (and usually a text prompt) and generates text. Typical stack: a vision encoder turns the image into a sequence of vectors, a projector maps those vectors into the language model's embedding space, then an [instruct](/ai-glossary/instruct-model/) LLM continues the sequence with next-[token](/ai-glossary/token/) prediction.

### What it outputs
Captions, answers to visual questions, OCR-style reads, or structured text such as JSON boxes. The boxes are still **text tokens** (digits and punctuation), not a separate detection head like YOLO.

### Related
[Multimodal](/ai-glossary/multimodal/) is the broader class (image, audio, video, text). A VLM is the common image+text instance used at [inference](/ai-glossary/inference/) for chat-over-pixels.
