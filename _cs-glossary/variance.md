---
title: Variance
synonyms:
  - Var
  - Population variance
keywords: variance, Var(X), standard deviation, MSE, statistics
related:
  - Expectation
  - Standard deviation
  - Covariance
  - Sample variance
  - Mean squared error
---
**Variance** measures spread: $\mathrm{Var}(X)=E[(X-E[X])^2]=E[X^2]-(E[X])^2$. Scaling: $\mathrm{Var}(aX+b)=a^2\mathrm{Var}(X)$. Standard deviation is $\sigma=\sqrt{\mathrm{Var}(X)}$.

For estimators, variance enters [MSE](/cs-glossary/mean-squared-error/) $=\mathrm{Var}+\mathrm{Bias}^2$. The unbiased sample variance uses divisor $n-1$; the [MLE](/cs-glossary/maximum-likelihood/) for Normal $\sigma^2$ uses $n$ and is biased but consistent.
