---
title: Knowledge Distillation
synonyms:
  - Distillation
related:
  - Quantization
  - Fine-Tuning
  - Softmax
---
**Knowledge distillation** trains a smaller student to mimic a larger teacher’s outputs or hidden states, compressing capability into a cheaper model. Classic forms match soft [softmax](/ai-glossary/softmax/) probabilities; others match logits, attention maps, or intermediate features.

### Why use it
- Lower latency and cost than serving the teacher
- Transfer behavior that is hard to label directly (style, tool use patterns)
- Complements [quantization](/ai-glossary/quantization/) and pruning in the efficiency toolbox

### LLM variants
Sequence-level distillation samples teacher text and trains the student with standard LM loss; on-policy variants reduce exposure bias. Distillation is not free quality—capacity gaps and domain mismatch still show up on hard evals.

Prefer a clear teacher, matched tokenizers when possible, and task metrics—not only matching train loss to the teacher.
