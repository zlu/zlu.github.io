---
title: Tensor
---
In machine learning (ML), a tensor is a generalization of scalars, vectors, and matrices to higher dimensions and is a core data structure used to represent and process data.

Formal Definition:

A tensor is a multidimensional array of numerical values. Its rank (or order) denotes the number of dimensions:
- 0D tensor: Scalar (e.g., 5)
- 1D tensor: Vector (e.g., \[1, 2, 3])
- 2D tensor: Matrix (e.g., \[[1, 2], \[3, 4]])
- 3D+ tensor: Higher-dimensional arrays (e.g., a stack of matrices)

Why Tensors Matter in ML:
- Input/output representation: Data like images (3D: height × width × channels), text sequences (2D: batch × sequence length), and time series are commonly represented as tensors.
- Efficient computation: Libraries like PyTorch and TensorFlow use tensor operations heavily, leveraging GPUs/TPUs for fast computation.
- Backpropagation: Tensors support automatic differentiation, essential for training neural networks.

Example in Code (PyTorch):
```python
import torch

# 2D tensor (matrix)
x = torch.tensor(\[[1.0, 2.0], \[3.0, 4.0]])
print(x.shape)  # torch.Size(\[2, 2])
```
In summary, a tensor is the fundamental building block for data in machine learning frameworks, offering a consistent and optimized structure for mathematical operations.
