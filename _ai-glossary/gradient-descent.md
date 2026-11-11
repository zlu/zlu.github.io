---
title: Gradient Descent
synonyms:
  - GD
related:
  - Stochastic Gradient Descent
  - Optimization
  - Optimizer
  - Learning Rate
  - Backpropagation
---
**Gradient descent** minimizes a scalar [loss](/ai-glossary/loss-function/) by repeatedly stepping parameters opposite the gradient: $\theta \leftarrow \theta - \eta \nabla_\theta L$. The [learning rate](/ai-glossary/learning-rate/) $\eta$ controls step size. Gradients come from [backpropagation](/ai-glossary/backpropagation/) in neural nets.

### Why it matters
- Core training loop behind almost all deep models
- Full-batch GD is rarely used at LLM scale; [SGD](/ai-glossary/stochastic-gradient-descent/) / mini-batches dominate
- Learning-rate schedules and adaptive methods build on this update rule

### How it shows up
Plain GD uses the full dataset each step. Practice uses mini-batches, momentum, and adaptive [optimizers](/ai-glossary/optimizer/) such as [Adam](/ai-glossary/adam/). Too-large $\eta$ diverges; too-small wastes compute. Plateauing loss often means LR, data, or capacity issues—not “gradient descent failed.”

Related: [gradient accumulation](/ai-glossary/gradient-accumulation/) and [batch size](/ai-glossary/batch-size/) control the effective step statistics when memory is tight.
