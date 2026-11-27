---
title: Conditional pmf / pdf
synonyms:
  - Conditional density
  - Conditional mass function
keywords: conditional pmf, conditional pdf, conditional distribution, statistics glossary, probability
related:
  - Conditional probability
  - Joint distribution
  - Marginal distribution
  - Conditional expectation
---
The **conditional pmf/pdf** of $X_1$ given $X_2=x_2,\ldots,X_k=x_k$ is the joint divided by the marginal of the conditioning variables:

$$
f_{X_1\mid X_2,\ldots,X_k}(x_1\mid x_2,\ldots,x_k)
=\frac{f_{X_1,\ldots,X_k}(x_1,\ldots,x_k)}{f_{X_2,\ldots,X_k}(x_2,\ldots,x_k)}.
$$

[Conditional expectation](/cs-glossary/conditional-expectation/) is the mean of this conditional distribution. Foundation of regression as $E[Y\mid X]$.
