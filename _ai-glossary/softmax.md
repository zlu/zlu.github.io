---
title: Softmax
---

## Softmax

**Softmax** is an activation function used in **multi-class classification** to convert a vector of real-valued scores (logits) into a **probability distribution** over classes.

---

### Definition
Given logits $z = (z_1, z_2, \dots, z_K)$,

$$
\text{softmax}(z_i)
= \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}},
\quad i = 1,\dots,K
$$

* Logits are the raw, unnormalized outputs of a model before an activation function (such as sigmoid or softmax) is applied.
---

### Key properties
- Outputs values in **(0, 1)**
- Probabilities **sum to 1**
- Preserves ordering: larger $z_i$ ⇒ larger probability
- Smooth and differentiable (good for backpropagation)

---

### Intuition
- Each $z_i$ is a **score** for class $i$
- Exponentiation emphasizes larger scores
- Normalization forces competition between classes
- Produces a probability-like output

Example:
$$
z = (2, 1, 0)
\;\Rightarrow\;
\text{softmax}(z) \approx (0.67,\;0.24,\;0.09)
$$

---

### Why softmax is used
- Enables **multi-class classification**
- Works naturally with **cross-entropy loss**
- Output can be interpreted as **class probabilities**

---

### Decision rule
$$
\hat{y} = \arg\max_i \text{softmax}(z_i)
$$
(Note: this is equivalent to $\arg\max_i z_i$.)

---

### Softmax vs Sigmoid

| Sigmoid | Softmax |
|-------|--------|
| Binary classification | Multi-class classification |
| Single output | Multiple outputs |
| Independent probabilities | Competing probabilities |
| $\sigma(z)\in(0,1)$ | $\sum_i p_i = 1$ |
