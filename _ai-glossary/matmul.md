---
title: Matmul
synonyms:
  - Matrix Multiplication
  - Matrix Multiply
  - GEMM
related:
  - Tensor
  - Softmax
  - Self-Attention
  - MLP
  - Flash Attention
---
A **matmul** (**matrix multiplication**, GEMM) multiplies two matrices (or a matrix and a vector): entry $(i,j)$ of $AB$ is the dot product of row $i$ of $A$ with column $j$ of $B$.

### Why it dominates LLM compute
Almost all transformer FLOPs are matmuls:
- $QK^\top$ scores and attention weights $\times V$ in [self-attention](/ai-glossary/self-attention/)
- Projection matrices $W_Q, W_K, W_V, W_O$
- Position-wise [MLP](/ai-glossary/mlp/) / FFN layers

Libraries expose it as `@` / `torch.matmul` / `np.matmul`. GPUs and TPUs are built around large GEMMs; [Flash Attention](/ai-glossary/flash-attention/) and fused kernels mainly reduce memory traffic around these ops. Shape and layout (batch, heads, seq) decide whether a matmul is compute-bound or memory-bound.
