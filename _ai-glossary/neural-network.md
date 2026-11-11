---
title: Neural Network
synonyms:
  - Neural Net
  - NN
related:
  - Deep Learning
  - Activation Function
  - Backpropagation
  - Parameter
  - Loss Function
---
A **neural network** is a parameterized function built from layered transformations — usually linear maps plus nonlinear [activations](/ai-glossary/activation-function/) — trained to minimize a [loss](/ai-glossary/loss-function/) via gradients ([backpropagation](/ai-glossary/backpropagation/)).

### Structure
- **Parameters** — weights and biases learned from data
- **Depth / width** — how many layers and units; “deep” stacks are [deep learning](/ai-glossary/deep-learning/)
- **Architecture** — MLPs, CNNs, RNNs, transformers; inductive bias matches the data

### Practice
Choose architecture and capacity for the task, regularize against overfitting, and validate out of sample. Modern vision, speech, and language systems — including transformers — are neural nets at scale; classical models still compete when data is tabular and limited.
