---
title: Pretraining
related:
  - Continued Pretraining
  - Self-Supervised Learning
  - Causal Language Model
  - Masked Model
  - Scaling Laws
  - Base Model
---
**Pretraining** is the large-scale initial training phase where a model learns general representations from massive unlabeled (or weakly labeled) data — e.g., [next-token prediction](/ai-glossary/next-token-prediction/) for [causal LMs](/ai-glossary/causal-language-model/) or masked LM for [masked models](/ai-glossary/masked-model/).

### What it buys you
A reusable [base model](/ai-glossary/base-model/) with broad language (and sometimes multimodal) competence. Downstream quality usually depends more on pretraining data mix, scale, and compute ([scaling laws](/ai-glossary/scaling-laws/)) than on any single fine-tuning trick.

### After pretraining
[Continued pretraining](/ai-glossary/continued-pretraining/) is a smaller, targeted second stage on your corpus. Instruction SFT and preference/[alignment](/ai-glossary/alignment/) then shape behavior for products.
