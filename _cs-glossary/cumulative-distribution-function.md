---
title: Cumulative distribution function
synonyms:
  - CDF
  - Distribution function
keywords: cdf, cumulative distribution function, F(x), statistics glossary
related:
  - Probability density function
  - Probability mass function
  - Convergence in distribution
  - Quantile
---
The **cumulative distribution function** (cdf) of a [random variable](/cs-glossary/random-variable/) $X$ is $F(x)=P(X\le x)$. It is nondecreasing, right-continuous, with limits $0$ and $1$ at $\pm\infty$.

For continuous $X$ with [pdf](/cs-glossary/probability-density-function/) $f$, $F'(x)=f(x)$ at continuity points. For discrete $X$, $F$ jumps by the [pmf](/cs-glossary/probability-mass-function/) mass at each point. [Convergence in distribution](/cs-glossary/convergence-in-distribution/) is defined via cdf convergence.
