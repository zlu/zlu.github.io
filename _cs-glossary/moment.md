---
title: Moment
synonyms:
  - Raw moment
  - Population moment
  - Statistical moment
keywords: moment, statistical moment, E[X^j], method of moments, statistics glossary, probability
related:
  - Method of moments
  - Expectation
  - Variance
  - Sample mean
---
A **moment** of a [random variable](/cs-glossary/random-variable/) $X$ is an expectation of a power: the $j$-th raw moment is $\mu'_j=E[X^j]$. The first moment is the mean; the second relates to [variance](/cs-glossary/variance/) via $\mathrm{Var}(X)=E[X^2]-(E[X])^2$. Central moments use $E[(X-\mu)^j]$.

[Method of moments](/cs-glossary/method-of-moments/) matches population moments $\mu'_j(\theta)$ to sample moments $m'_j=n^{-1}\sum x_i^j$. In time series and method-of-moments estimators, “moments” means these expectations — not physical torque.
