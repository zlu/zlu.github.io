---
title: Detection Head
synonyms:
  - Object Detection Head
  - Box Head
related:
  - Tensor
  - Convolutional Neural Network
  - Vision-Language Model
---
A **detection head** is the output module of a classical object detector (YOLO, RetinaNet, Faster R-CNN). It predicts box geometry and class scores as [tensors](/ai-glossary/tensor/), usually followed by non-maximum suppression (NMS). Training uses localization and classification losses on those tensors, not next-token cross-entropy on a JSON string.

### Contrast with VLM grounding
A [vision-language model](/ai-glossary/vision-language-model/) that “returns boxes” still emits **text** (digits and punctuation). There is no detection head. Parsing the string and drawing rectangles is post-processing outside the model.
