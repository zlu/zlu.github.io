---
title: Sample variance
synonyms:
  - Empirical variance
  - S-squared
keywords: sample variance, S^2, n-1, unbiased variance, statistics
related:
  - Variance
  - Sample mean
  - Degrees of freedom
  - Unbiased estimator
  - Maximum likelihood
---
The **sample variance** with Bessel correction is

$$
S^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\bar X)^2.
$$

The $n-1$ [degrees of freedom](/cs-glossary/degrees-of-freedom/) account for estimating $\mu$ by $\bar X$; $S^2$ is [unbiased](/cs-glossary/unbiased-estimator/) for $\sigma^2$ in the iid finite-variance case. The MoM/MLE version divides by $n$ and is biased low by $\sigma^2/n$ under Normality.
