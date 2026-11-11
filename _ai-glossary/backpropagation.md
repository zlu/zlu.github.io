---
title: Backpropagation
synonyms:
  - Backprop
related:
  - Gradient Descent
  - Neural Network
  - Loss Function
  - Optimizer
---
**Backpropagation** computes gradients of the [loss](/ai-glossary/loss-function/) w.r.t. each parameter by applying the chain rule from output to input through the computation graph. Those gradients drive [optimizers](/ai-glossary/optimizer/) such as [SGD](/ai-glossary/stochastic-gradient-descent/) or [Adam](/ai-glossary/adam/) in the [gradient descent](/ai-glossary/gradient-descent/) update.

### Why it matters
- Workhorse training algorithm for deep [neural networks](/ai-glossary/neural-network/), including Transformers
- Automatic differentiation frameworks implement it; you mostly design the forward graph and loss
- Memory for activations (and tricks like [gradient checkpointing](/ai-glossary/gradient-checkpointing/)) often dominates training cost

### How it shows up
Forward pass → scalar loss → backward pass fills `.grad` → optimizer step → zero grads. Vanishing/exploding gradients historically limited depth; residuals, norms, and careful init mitigate that. Bugs that “train but don’t learn” are often stopped gradients, wrong loss reduction, or accidentally detached tensors—not mysterious backprop failures.
