---
title: Method of moments
synonyms:
  - MoM
  - Moment matching
  - Moment estimator
keywords: method of moments, MoM, moment matching, estimator, statistics glossary, probability
related:
  - Moment
  - Maximum likelihood
  - Estimator
  - Sample mean
  - Consistency
---
**Method of moments** (MoM) estimates parameters by equating population moments to sample moments and solving for $\theta$:

$$
\mu'_j(\theta)=m'_j=n^{-1}\sum_{i=1}^n x_i^j,\quad j=1,\ldots,k.
$$

Examples: Normal $\hat\mu=\bar X$, $\hat\sigma^2_{\mathrm{MoM}}=n^{-1}\sum(X_i-\bar X)^2$; Poisson $\hat\lambda=\bar X$. Easy to compute; not always optimal versus [MLE](/cs-glossary/maximum-likelihood/).
