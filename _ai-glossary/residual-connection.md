---
title: Residual Connection
synonyms:
  - Skip Connection
related:
  - Transformer
  - Layer Normalization
---
A **residual (skip) connection** adds a block’s input to its output: $x + F(x)$. This eases gradient flow and enables very deep networks — ResNets and every Transformer block use them.
