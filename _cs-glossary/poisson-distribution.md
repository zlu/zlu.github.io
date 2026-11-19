---
title: Poisson distribution
synonyms:
  - Poisson
  - Poisson(lambda)
keywords: Poisson distribution, Poisson lambda, count data, MLE, statistics
related:
  - Bernoulli distribution
  - Exponential distribution
  - Method of moments
  - Maximum likelihood
  - Sample mean
---
A **Poisson**($\lambda$) [random variable](/cs-glossary/random-variable/) counts rare events in a fixed interval: $P(X=k)=e^{-\lambda}\lambda^k/k!$ for $k=0,1,2,\ldots$. Mean and variance both equal $\lambda$.

For an [iid](/cs-glossary/iid/) sample, [MoM](/cs-glossary/method-of-moments/) and [MLE](/cs-glossary/maximum-likelihood/) both give $\hat\lambda=\bar X$. Support is the non-negative integers.
