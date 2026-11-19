---
title: Probability density function
synonyms:
  - PDF
  - Density function
keywords: pdf, probability density function, continuous distribution, statistics
related:
  - Probability mass function
  - Cumulative distribution function
  - Normal distribution
  - Random variable
---
A **probability density function** (pdf) $f$ describes a continuous [random variable](/cs-glossary/random-variable/): $f(x)\ge 0$ and $\int_{-\infty}^{\infty}f(x)\,dx=1$. Probabilities are areas: $P(a\le X\le b)=\int_a^b f$. Point probabilities are zero: $P(X=x)=0$.

The [cdf](/cs-glossary/cumulative-distribution-function/) is $F(x)=\int_{-\infty}^x f$. For the [Normal](/cs-glossary/normal-distribution/), read $\mu$ and $\sigma$ from the formula coefficients rather than integrating by hand.
