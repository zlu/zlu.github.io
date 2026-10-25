---
title: Matmul
synonyms:
  - Matrix Multiplication
  - Matrix Multiply
related:
  - Tensor
  - Softmax
  - Self-Attention
  - MLP
---
A **matmul** (**matrix multiplication**) multiplies two matrices (or a matrix and a vector) with the usual linear-algebra rule: entry $(i,j)$ of $AB$ is the dot product of row $i$ of $A$ with column $j$ of $B$.

In attention code you see matmuls constantly—$QK^\top$ for scores, attention weights times $V$ for the context, and the $W_Q, W_K, W_V$ projections. Libraries expose it as `@` / `torch.matmul` / `np.matmul`; GPUs are built to run large matmuls fast.
